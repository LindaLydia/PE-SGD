from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import EarlyStoppingCallback
from typing import Optional
from datasets import Dataset
import accelerate
import copy

import json
import numpy as np
import pandas as pd
import torch

from .metric_logging_plotting import logging_plotting
from peft import LoraConfig, get_peft_model, TaskType
from .instruction_addition import chat_template_tokenize_example
from .fine_tuned_model_eval import next_token_accuracy
from .trainer import PerSampleLossTrainer
from .trainer_image import PerSampleLossTrainerImage
from .image_dataset_processing import image_preprocess



def sft_fine_tune(
    model,
    tokenizer,
    train_dataset,
    output_dir,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_steps=500,
    logging_steps=5,
    learning_rate=5e-5,
    greater_is_better=False,
    remove_unused_columns=False,
    add_instruction=False,
    instruction=None,
    eval_dataset= None,
    gradient_accumulation_steps=1,
):
    if add_instruction == False:
        instruction = None
        
    # data_collator = DataCollatorForLanguageModeling(
    #     tokenizer=tokenizer, mlm=False
    # )
    if eval_dataset is not None:
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            eval_strategy ="epoch",
            # eval_steps=logging_steps,  # run eval every <save_steps> steps
            save_steps=save_steps,        # also save every <save_steps> steps
            logging_steps=logging_steps,
            learning_rate=learning_rate,
            warmup_ratio=0.08,
            weight_decay=0.01,
            prediction_loss_only=True,
            report_to=[],
            remove_unused_columns=False,
        )
    else:
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            save_steps=save_steps,        # also save every <save_steps> steps
            logging_steps=logging_steps,
            learning_rate=learning_rate,
            warmup_ratio=0.08,
            weight_decay=0.01,
            prediction_loss_only=True,
            report_to=[],
            remove_unused_columns=False,
        )


    if 'ImageProcessor' in tokenizer.__class__.__name__:
        # Encode the training dataset if not already tokenized
        if not ("pixel_values" in train_dataset[0]):
            train_dataset = train_dataset.map(image_preprocess, fn_kwargs={"processor": tokenizer})
            train_dataset.set_format(type='torch', columns=['pixel_values', 'labels'])
        if eval_dataset is not None and not ("pixel_values" in eval_dataset[0]):
            eval_dataset = eval_dataset.map(image_preprocess, fn_kwargs={"processor": tokenizer})
            eval_dataset.set_format(type='torch', columns=['pixel_values', 'labels'])
    else:
        # Encode the evaluation dataset if not already tokenized
        if not ("input_ids" in train_dataset[0]):
            if not add_instruction:
                train_dataset = tokenizer(train_dataset['text'], truncation=True, padding=True, return_tensors="pt")
                train_dataset["labels"] = train_dataset["input_ids"].clone()
                # Convert to HuggingFace Dataset if needed
                train_dataset = Dataset.from_dict(train_dataset)
            else:
                assert instruction != None, f'[ERROR] When specifying add_instruction=True, instruction should be provided.'
                print(f"{type(train_dataset)=}")
                print(f"{train_dataset.column_names=}")
                train_dataset = train_dataset.map(
                                    chat_template_tokenize_example, 
                                    fn_kwargs={"prompt_config": instruction, "tokenizer": tokenizer, "max_length": 2048}, 
                                    remove_columns=train_dataset.column_names
                                ) #, remove_columns=train_dataset["train"].column_names
                train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
                # train_dataset = Dataset.from_dict(train_dataset)
        if eval_dataset is not None and not ("input_ids" in eval_dataset[0]):
            if not add_instruction:
                # print(f"[debugging] eval_dataset={eval_dataset}")
                eval_dataset = tokenizer(eval_dataset['text'], truncation=True, padding=True, return_tensors="pt")
                eval_dataset["labels"] = eval_dataset["input_ids"].clone()
                # Convert to HuggingFace Dataset if needed
                eval_dataset = Dataset.from_dict(eval_dataset)
            else:
                assert instruction != None, f'[ERROR] When specifying add_instruction=True, instruction should be provided.'
                print(f"{type(eval_dataset)=}")
                print(f"{eval_dataset.column_names=}")
                eval_dataset = eval_dataset.map(
                                    chat_template_tokenize_example, 
                                    fn_kwargs={"prompt_config": instruction, "tokenizer": tokenizer, "max_length": 2048}, 
                                    remove_columns=eval_dataset.column_names
                                ) #, remove_columns=eval_dataset["train"].column_names
                eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
                # eval_dataset = Dataset.from_dict(eval_dataset)

    if 'ImageProcessor' in tokenizer.__class__.__name__:
        # trainer = Trainer(
        trainer = PerSampleLossTrainerImage(
            model=model,
            args=training_args,
            # data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if (eval_dataset is not None) else train_dataset,  # Use the same dataset for testing
            # compute_metrics=next_token_accuracy,
        )
    else:
        # trainer = Trainer(
        trainer = PerSampleLossTrainer(
            model=model,
            args=training_args,
            # data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if (eval_dataset is not None) else train_dataset,  # Use the same dataset for testing
            # compute_metrics=next_token_accuracy,
        )
    
    _accelerator = accelerate.Accelerator()
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
    eval_result = trainer.evaluate(eval_dataset, metric_key_prefix="eval") if eval_dataset is not None else None
    # eval_result = None

    model, offload_hook = accelerate.cpu_offload_with_hook(model, execution_device="cuda")
    offload_hook.offload()

    return eval_result, trainer.model, tokenizer


def sft_fine_tune_until_converge(
    model,
    tokenizer,
    train_dataset,
    output_dir,
    per_device_train_batch_size=2,
    min_delta=0.001,
    patience=2,
    save_steps=500,
    logging_steps=100,
    learning_rate=5e-5,
    max_epochs=20,
    add_instruction=False,
    instruction=None,
):
    if add_instruction == False:
        instruction = None
    
    """
    Fine-tune until convergence using early stopping on training loss.
    """
    # data_collator = DataCollatorForLanguageModeling(
    #     tokenizer=tokenizer, mlm=False
    # )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=max_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        warmup_ratio=0.08,
        weight_decay=0.01,
        prediction_loss_only=True,
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        remove_unused_columns=False,
    )
    # Encode the training dataset if not already tokenized
    if not ("input_ids" in train_dataset[0]):
        if not add_instruction:
            # print(f"[debugging] train_dataset={train_dataset}")
            train_dataset = tokenizer(train_dataset['text'], truncation=True, padding=True, return_tensors="pt")
            train_dataset["labels"] = train_dataset["input_ids"].clone()
            # Convert to HuggingFace Dataset if needed
            train_dataset = Dataset.from_dict(train_dataset)
        else:
            assert instruction != None, f'[ERROR] When specifying add_instruction=True, instruction should be provided.'
            print(f"{type(train_dataset)=}")
            print(f"{train_dataset.column_names=}")
            train_dataset = train_dataset.map(
                chat_template_tokenize_example, 
                fn_kwargs={"prompt_config": instruction, "tokenizer": tokenizer, "max_length": 2048}, 
                remove_columns=dataset.column_names) #, remove_columns=train_dataset["train"].column_names
            train_dataset = Dataset.from_dict(train_dataset)

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=patience,
        early_stopping_threshold=min_delta,
    )

    # trainer = Trainer(
    trainer = PerSampleLossTrainer(
        model=model,
        args=training_args,
        # data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,  # Use the same dataset for testing
        callbacks=[early_stopping],
    )
    
    _accelerator = accelerate.Accelerator()
    # model = accelerate.dispatch_model(model, device_map='auto')
    model, trainer = _accelerator.prepare(model, trainer)
    model.train()

    trainer.train()
    trainer.model.save_pretrained(output_dir)
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

    eval_result = trainer.evaluate(train_dataset, metric_key_prefix="train")

    model, offload_hook = accelerate.cpu_offload_with_hook(model, execution_device="cuda")
    offload_hook.offload()

    return eval_result, trainer.model, tokenizer



    # def sft_fine_tune_lora(
    #     model,
    #     tokenizer,
    #     train_dataset,
    #     output_dir,
    #     per_device_train_batch_size=2,
    #     num_train_epochs=3,
    #     save_steps=500,
    #     logging_steps=5,
    #     learning_rate=5e-5,
    #     lora_r=8,
    #     lora_alpha=16,
    #     lora_dropout=0.1,
    #     target_modules=None,
    # ):
    #     """
    #     Fine-tune using LoRA (Low-Rank Adaptation).
    #     """
    #     lora_config = LoraConfig(
    #         r=lora_r,
    #         lora_alpha=lora_alpha,
    #         target_modules=target_modules,
    #         lora_dropout=lora_dropout,
    #         bias="none",
    #         task_type=TaskType.CAUSAL_LM,
    #     )
    #     model = get_peft_model(model, lora_config)

    #     data_collator = DataCollatorForLanguageModeling(
    #         tokenizer=tokenizer, mlm=False
    #     )
    #     training_args = TrainingArguments(
    #         output_dir=output_dir,
    #         overwrite_output_dir=True,
    #         num_train_epochs=num_train_epochs,
    #         per_device_train_batch_size=per_device_train_batch_size,
    #         save_steps=save_steps,
    #         logging_steps=logging_steps,
    #         learning_rate=learning_rate,
    #         prediction_loss_only=True,
    #         report_to=[],
    #         remove_unused_columns=False,
    #     )
    #     if not ("input_ids" in train_dataset[0]):
    #         train_dataset = tokenizer(train_dataset['text'], truncation=True, padding=True, return_tensors="pt")
    #         train_dataset = Dataset.from_dict(train_dataset)

    #     # trainer = Trainer(
    #     trainer = PerSampleLossTrainer(
    #         model=model,
    #         args=training_args,
    #         data_collator=data_collator,
    #         train_dataset=train_dataset,
    #     )

    #     trainer.train()
    #     trainer.save_model(output_dir)
    #     tokenizer.save_pretrained(output_dir)

    #     training_loss_history = []
    #     train_log = pd.DataFrame(trainer.state.log_history)
    #     for log in trainer.state.log_history:
    #         if "train_loss" in log and "step" in log:
    #             training_loss_history.append({
    #                 "step": log.get("step", None),
    #                 "loss": log.get("train_loss", None)
    #             })
    #         elif 'loss' in log and "step" in log:
    #             training_loss_history.append({
    #                 "step": log.get("step", None),
    #                 "loss": log.get("loss", None)
    #             })
    #     with open(f"{output_dir}/training_loss_history.json", "w") as f:
    #         json.dump(training_loss_history, f, indent=2)
    #     logging_plotting(training_loss_history, output_dir)

    #     return trainer.evaluate(train_dataset, metric_key_prefix="train"), trainer.model, tokenizer