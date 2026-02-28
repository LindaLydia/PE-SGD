from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import EarlyStoppingCallback
from typing import Optional
from datasets import Dataset

import json
import pandas as pd
import torch
import accelerate

from .metric_logging_plotting import logging_plotting
from peft import LoraConfig, get_peft_model, TaskType

class WeightedTrainer(Trainer):
    def __init__(self, *args, weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = weight

    def compute_loss(self, model, inputs, num_items_in_batch=8, return_outputs=False):
        labels = inputs.get("labels")
        if labels is not None:
            inputs["labels"] = labels.to(next(model.parameters()).device)
        # print(f"[debugging] {inputs.keys()=}")
        # print(f"[debugging] {inputs=}")
        outputs = model(**inputs)
        # # loss = outputs.loss
        # print(f"[debugging] {outputs.logits.shape=}")

        # Calculate per-sample loss for next token prediction
        # outputs.logits: [batch_size, seq_len, vocab_size]
        # inputs["labels"]: [batch_size, seq_len]
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = inputs["labels"][..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        # Flatten the tokens for loss computation
        loss_per_token = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        # Reshape to [batch_size, seq_len-1]
        loss_per_token = loss_per_token.view(shift_labels.size())
        # Mask out padding tokens
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            shift_attention_mask = attention_mask[..., 1:].contiguous()
            loss_per_token = loss_per_token * shift_attention_mask
            lengths = shift_attention_mask.sum(dim=1)
        else:
            lengths = torch.ones(loss_per_token.size(0), device=loss_per_token.device) * loss_per_token.size(1)
        # Per-sample loss: sum over tokens, divide by number of valid tokens
        loss = loss_per_token.sum(dim=1) / lengths

        weight = inputs.get("reward", torch.tensor([1.0/loss.size(0)] * loss.size(0), device=loss.device))

        # print(f"[debugging] {loss=}, {loss.shape=}")
        # # print(f"[debugging] {outputs=}")
        # print(f"[debugging] {weight=}, {loss=}")
        # print(f"[debugging] {weight.shape=}, {loss.shape=}")
        loss = (loss * weight).mean()
        # print(f"[debugging] weighted {loss=}, original {outputs.loss=}")
        return (loss, outputs) if return_outputs else loss


def weighted_fine_tune(
    model,
    tokenizer,
    train_dataset,
    output_dir,
    weight=None,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_steps=500,
    logging_steps=5,
    learning_rate=5e-5,
):
    if weight == None:
        weight = torch.tensor([1.0/len(train_dataset)] * len(train_dataset)).to(next(model.parameters()).device)
    # print(f"[debugging] {weight=}")
    # print(f"[debugging] {len(train_dataset)=}, {train_dataset[0]=}")
    # print(F"[debugging] {train_dataset.column_names=}")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        prediction_loss_only=True,
        report_to=[],
        remove_unused_columns=False,
    )
    # Encode the training dataset if not already tokenized
    if not ("input_ids" in train_dataset[0]):
        # print(f"[debugging] train_dataset={train_dataset}")
        _train_dataset = tokenizer(train_dataset['text'], truncation=True, padding=True, return_tensors="pt")
        # Keep other columns in the dataset
        for col in train_dataset.column_names:
            # if col not in ["input_ids", "attention_mask", "labels"]:
            if col.lower() in ['reward', 'rewards']:  # Assuming 'text' is the column to tokenize
                # Add other columns to the tokenized dataset
                _train_dataset['reward'] = train_dataset[col]
        # Convert to HuggingFace Dataset if needed
        train_dataset = Dataset.from_dict(_train_dataset)
        # print(F"[debugging] {train_dataset.column_names=}")

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,  # Use the same dataset for testing
        weight=weight,
    )
    
    _accelerator = accelerate.Accelerator()
    # model = accelerate.dispatch_model(model, device_map='auto')
    # model, data_collator, train_dataset = _accelerator.prepare(model, data_collator, train_dataset)
    model, trainer = _accelerator.prepare(model, trainer)
    # trainer = _accelerator.prepare(trainer)
    model.train()

    trainer.train()
    trainer.model.save_pretrained(output_dir) # Save the lora part
    # trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save training loss as JSON
    training_loss_history = []
    # print(f"[debugging], see {trainer.state.log_history=}")
    train_log = pd.DataFrame(trainer.state.log_history)
    for log in trainer.state.log_history:
        if "train_loss" in log and "step" in log:
            training_loss_history.append({
                "step": log.get("step", None),
                "loss": log.get("train_loss", None)
            })
        elif 'loss' in log and "step" in log:
            training_loss_history.append({
                "step": log.get("step", None),
                "loss": log.get("loss", None)
            })
    with open(f"{output_dir}/training_loss_history.json", "w") as f:
        json.dump(training_loss_history, f, indent=2)
    logging_plotting(training_loss_history, output_dir)

    # TODO: bug, evaluation of multiple GPUs does not work properly
    # eval_result = trainer.evaluate(train_dataset, metric_key_prefix="train")
    eval_result = None

    # model.to('cpu')  # Move model back to CPU after training
    model, offload_hook = accelerate.cpu_offload_with_hook(model, execution_device="cuda")
    offload_hook.offload()

    return eval_result, trainer.model, tokenizer

