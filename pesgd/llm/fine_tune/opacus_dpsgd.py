import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import EarlyStoppingCallback
from typing import Optional
from datasets import Dataset
import accelerate
import copy
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

import json
import pandas as pd
import gc
import tqdm
import numpy as np

from .metric_logging_plotting import logging_plotting
from peft import LoraConfig, get_peft_model, TaskType
from .instruction_addition import chat_template_tokenize_example
from .fine_tuned_model_eval import evaluate_model_on_private_data, next_token_accuracy
from .trainer import per_sample_loss_function
from .trainer_image import per_sample_loss_function_image
from .image_dataset_processing import image_preprocess
from pe.logging import execution_logger
import os


def opacus_dpsgd_fine_tune(
    model,
    tokenizer,
    train_dataset,
    output_dir,
    per_device_train_batch_size=2,
    num_train_epochs=10, # should match the pe-iteration number
    save_steps=500,
    logging_steps=5,
    learning_rate=5e-5,
    epsilon=1.0,
    delta=1e-5,
    val_sample_ratio=0.2, # sampling ratio for dp-sgd gradient calculation
    greater_is_better=False,
    remove_unused_columns=False,
    add_instruction=False,
    instruction=None,
    eval_dataset= None,
    gradient_accumulation_steps=1,
    stop_one_batch=False,
):
    if add_instruction == False:
        instruction = None
        
    # data_collator = DataCollatorForLanguageModeling(
    #     tokenizer=tokenizer, mlm=False
    # )
    # if eval_dataset is not None:
    #     training_args = TrainingArguments(
    #         output_dir=output_dir,
    #         overwrite_output_dir=True,
    #         num_train_epochs=num_train_epochs,
    #         per_device_train_batch_size=per_device_train_batch_size,
    #         gradient_accumulation_steps=gradient_accumulation_steps,
    #         eval_strategy ="epoch",
    #         # eval_steps=logging_steps,  # run eval every <save_steps> steps
    #         save_steps=save_steps,        # also save every <save_steps> steps
    #         logging_steps=logging_steps,
    #         learning_rate=learning_rate,
    #         warmup_ratio=0.08,
    #         weight_decay=0.01,
    #         prediction_loss_only=True,
    #         report_to=[],
    #         remove_unused_columns=False,
    #     )
    # else:
    #     training_args = TrainingArguments(
    #         output_dir=output_dir,
    #         overwrite_output_dir=True,
    #         num_train_epochs=num_train_epochs,
    #         per_device_train_batch_size=per_device_train_batch_size,
    #         save_steps=save_steps,        # also save every <save_steps> steps
    #         logging_steps=logging_steps,
    #         learning_rate=learning_rate,
    #         warmup_ratio=0.08,
    #         weight_decay=0.01,
    #         prediction_loss_only=True,
    #         report_to=[],
    #         remove_unused_columns=False,
    #     )
    if 'ImageProcessor' in tokenizer.__class__.__name__:
        # Encode the training dataset if not already tokenized
        if not ("pixel_values" in train_dataset[0]):
            train_dataset = train_dataset.map(image_preprocess, fn_kwargs={"processor": tokenizer})
            train_dataset.set_format(type='torch', columns=['pixel_values', 'labels'])
        if eval_dataset is not None and not ("pixel_values" in eval_dataset[0]):
            eval_dataset = eval_dataset.map(image_preprocess, fn_kwargs={"processor": tokenizer})
            eval_dataset.set_format(type='torch', columns=['pixel_values', 'labels'])
    else:
        # Encode the training dataset if not already tokenized
        if not ("input_ids" in train_dataset[0]):
            if not add_instruction:
                # print(f"[debugging] train_dataset={train_dataset}")
                train_dataset = tokenizer(train_dataset['text'], truncation=True, padding=True, return_tensors="pt")
                train_dataset["labels"] = train_dataset["input_ids"].clone()
                # Convert to HuggingFace Dataset
                train_dataset = Dataset.from_dict(train_dataset)
                train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
            else:
                assert instruction != None, f'[ERROR] When specifying add_instruction=True, instruction should be provided.'
                train_dataset = train_dataset.map(
                                    chat_template_tokenize_example, 
                                    fn_kwargs={"prompt_config": instruction, "tokenizer": tokenizer, "max_length": 2048}, 
                                    remove_columns=train_dataset.column_names
                                ) #, remove_columns=train_dataset["train"].column_names
                train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        if eval_dataset is not None and not ("input_ids" in eval_dataset[0]):
            if not add_instruction:
                # print(f"[debugging] eval_dataset={eval_dataset}")
                eval_dataset = tokenizer(eval_dataset['text'], truncation=True, padding=True, return_tensors="pt")
                eval_dataset["labels"] = eval_dataset["input_ids"].clone()
                # Convert to HuggingFace Dataset
                eval_dataset = Dataset.from_dict(eval_dataset)
                eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
            else:
                assert instruction != None, f'[ERROR] When specifying add_instruction=True, instruction should be provided.'
                eval_dataset = eval_dataset.map(
                                    chat_template_tokenize_example, 
                                    fn_kwargs={"prompt_config": instruction, "tokenizer": tokenizer, "max_length": 2048}, 
                                    remove_columns=eval_dataset.column_names
                                ) #, remove_columns=eval_dataset["train"].column_names
                eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
                # print(f"[debugging] in <./pesgd/llm/fine_tune/opacus_sgd.py> {eval_dataset.column_names=}")

    if not 'ImageProcessor' in tokenizer.__class__.__name__:
        model.enable_input_require_grads()
    # model = GradSampleModule(model)
    model.train()
    MAX_GRAD_NORM = 1.0

    # Construct train DataLoader
    # train_dataloader = DataLoader(
    #     train_dataset,
    #     batch_size=per_device_train_batch_size, shuffle=True, collate_fn=data_collator  # collates dicts into tensors
    # )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=int(len(train_dataset)*val_sample_ratio), shuffle=True# , collate_fn=data_collator  # collates dicts into tensors
    )
    # test_dataloader = DataLoader(
    #     eval_dataset,
    #     batch_size=per_device_train_batch_size, shuffle=False, collate_fn=data_collator  # collates dicts into tensors
    # )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01) # TODO: warmup_ratio=0.08, should be used with a scheduler, but since we are doing a single step update, I don't know if this should take effect
    privacy_engine = PrivacyEngine()
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    if epsilon == 1E7:
        model, optimizer, train_dataloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_dataloader,
            noise_multiplier=0.0,
            epochs=num_train_epochs,
            max_grad_norm=MAX_GRAD_NORM,
        )
    else:
        model, optimizer, train_dataloader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_dataloader,
            target_delta=delta,
            target_epsilon=epsilon,
            epochs=num_train_epochs,
            max_grad_norm=MAX_GRAD_NORM,
        )
    # model, optimizer, criterion, train_dataloader = (
    #     privacy_engine.make_private_with_epsilon(
    #         module=model,
    #         optimizer=optimizer,
    #         data_loader=train_dataloader,
    #         criterion=criterion, # TODO: define criterion
    #         target_delta=delta,
    #         target_epsilon=epsilon,
    #         epochs=num_train_epochs,
    #         max_grad_norm=MAX_GRAD_NORM,
    #         grad_sample_mode="ghost",
    #     )
    # )

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     data_collator=data_collator,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset if (eval_dataset is not None) else train_dataset,  # Use the same dataset for testing
    #     compute_metrics=next_token_accuracy,
    # )
    torch.cuda.empty_cache()
    gc.collect()
    # print(f"[debugging] in <./pesgd/llm/fine_tune/opacus_sgd.py> 1: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    
    _accelerator = accelerate.Accelerator()
    # model, train_dataloader = _accelerator.prepare(model, train_dataloader)
    model, optimizer, criterion, train_dataloader = _accelerator.prepare(model, optimizer, criterion, train_dataloader)
    model.train()
    
    device = next(model.parameters()).device
    
    torch.cuda.empty_cache()
    gc.collect()
    # print(f"[debugging] in <./pesgd/llm/fine_tune/opacus_sgd.py> 2: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    training_loss_history = []
    step = 0
    device = next(model.parameters()).device
    for epoch in range(1, num_train_epochs + 1):
        losses = []
        log_counter = 0
    
        with BatchMemoryManager(
                data_loader=train_dataloader, max_physical_batch_size=per_device_train_batch_size, optimizer=optimizer.optimizer
            ) as memory_safe_data_loader:
            
            counter = 0
            for batch in tqdm.tqdm(memory_safe_data_loader):
                model.train()
                # print(f"[debugging] in <./ep/llm/fine_tune/dp_sgd.py> {epoch=}, {step=}, {batch['input_ids'].shape=}")
                optimizer.zero_grad()
                inputs = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**inputs)  # output = loss, logits, hidden_states, attentions
                # loss = outputs.loss
                if 'ImageProcessor' in tokenizer.__class__.__name__:
                    loss = per_sample_loss_function_image(inputs['labels'], outputs.logits, reduction='none').mean()
                else:
                    loss = per_sample_loss_function(inputs['labels'], outputs.logits, reduction='none').mean()
                # print(f"[debugging] in <./ep/llm/fine_tune/dp_sgd.py> {outputs.loss=}, real per_sample_everage_{loss=}")
                loss.backward()
                losses.append(loss.item())
                optimizer.step()

                counter += 1
                if stop_one_batch and counter >= int(len(train_dataset)*val_sample_ratio)/per_device_train_batch_size:
                    print(f"Stop with only one parameter update requirment, {counter=} should be no smaller than {int(len(train_dataset)*val_sample_ratio)=}/{per_device_train_batch_size=}={int(len(train_dataset)*val_sample_ratio)/per_device_train_batch_size}")
                    break
                
                if log_counter % save_steps == 0:
                    checkpoint_dir = os.path.join(output_dir, f'checkpoint-{log_counter}')
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    model._module.save_pretrained(checkpoint_dir) # Save the lora part
                    tokenizer.save_pretrained(checkpoint_dir)
                    torch.cuda.empty_cache()
                    gc.collect()
                log_counter += 1
                
                # # TODO: this is a very ugly way to break the loop and the privacy budget is not exploited totally
                # break

            # if step > 0 and (step % ((logging_steps*2))) == 0:
            train_loss = np.mean(losses)
            eval_loss, eval_accuracy = evaluate_model_on_private_data(model, tokenizer, eval_dataset, add_instruction=add_instruction, instruction=instruction)
            if (not stop_one_batch) and epsilon != 1E7:
                eps = privacy_engine.get_epsilon(delta)
            else:
                eps = -1.0
            training_loss_history.append({
                "epoch": epoch,
                "step": step,
                "loss": train_loss,
                "eval_loss": eval_loss,
                "eval_accuracy": eval_accuracy,
                "epsilon": eps,
            })
            print(
                f"Epoch: {epoch} | "
                f"Step: {step} | "
                f"Train loss: {train_loss:.3f} | "
                f"Eval loss: {eval_loss:.3f} | "
                f"Eval accuracy: {eval_accuracy:.3f} | "
                f"ɛ: {eps:.2f}"
            )
            execution_logger.info(
                f"Train on private train data, Epoch: {epoch} | "
                f"Step: {step} | "
                f"Train loss: {train_loss:.6f} | "
                f"evaluation on test private data - Loss: {eval_loss:.6f}, "
                f"Token Accuracy: {eval_accuracy:.6f} | "
                f"ɛ: {eps:.3f}"
            )
            step += 1
        
        torch.cuda.empty_cache()
        gc.collect()

    model._module.save_pretrained(output_dir) # Save the lora part
    tokenizer.save_pretrained(output_dir)

    # Save training loss as JSON
    with open(f"{output_dir}/training_loss_history.json", "w") as f:
        json.dump(training_loss_history, f, indent=2)
    logging_plotting(training_loss_history, output_dir)

    # # TODO: bug, evaluation of multiple GPUs does not work properly
    # # eval_result = trainer.evaluate(train_dataset, metric_key_prefix="train")
    # eval_result = trainer.evaluate(eval_dataset, metric_key_prefix="eval") if eval_dataset is not None else None
    eval_result = None

    model, offload_hook = accelerate.cpu_offload_with_hook(model, execution_device="cuda")
    offload_hook.offload()

    return eval_result, model._module, tokenizer

