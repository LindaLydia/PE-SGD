import torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers import EarlyStoppingCallback
from typing import Optional
from datasets import Dataset
import accelerate
from peft import LoraConfig, get_peft_model, TaskType

import os
import json
import pandas as pd
import gc

from .fine_tune.metric_logging_plotting import logging_plotting
from .ghostEngines.graddotprod_engine import GradDotProdEngine

def ghost_suite_grad_dot(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    output_dir,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_steps=500,
    logging_steps=5,
    learning_rate=5e-5
):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size//4,
        per_device_eval_batch_size=per_device_train_batch_size//4-1,
        # per_device_train_batch_size=8,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        prediction_loss_only=True,
        report_to=[],
        remove_unused_columns=False,
    )
    # print(f"[debugging] torch.cuda.memory, 1: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    # # Encode the training dataset if not already tokenized
    # if not ("input_ids" in train_dataset[0]):
    #     train_dataset = tokenizer(train_dataset['text'], truncation=True, padding=True, return_tensors="pt")
    #     # Convert to HuggingFace Dataset if needed
    #     train_dataset = Dataset.from_dict(train_dataset)
    # if not ("input_ids" in val_dataset[0]):
    #     val_dataset = tokenizer(val_dataset['text'], truncation=True, padding=True, return_tensors="pt")
    #     val_dataset = Dataset.from_dict(val_dataset)
    if not ("input_ids" in train_dataset[0]):
        # total_dataset = tokenizer(train_dataset['text']+val_dataset['text'], truncation=True, padding=True, return_tensors="pt")
        # train_dataset = {k: v[:len(train_dataset)] for k, v in total_dataset.items()}
        # val_dataset = {k: v[len(train_dataset):] for k, v in total_dataset.items()}
        total_dataset = tokenizer(train_dataset['text'][:8]+val_dataset['text'][8:], truncation=True, padding=True, return_tensors="pt")
        train_dataset = {k: v[:8] for k, v in total_dataset.items()}
        val_dataset = {k: v[8:] for k, v in total_dataset.items()}
        # print(f"[debugging] {val_dataset=}, {val_dataset['input_ids'].shape=}")
        train_dataset = Dataset.from_dict(train_dataset)
        val_dataset = Dataset.from_dict(val_dataset)
    # print(f"[debugging] torch.cuda.memory, 2: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # Use the same dataset for testing
    )
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         # param.requires_grad = True
    #         print(f"[debugging] Trainable parameter: {name}, shape: {param.shape}")
    # print(f"[debugging] torch.cuda.memory, 3: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    model.enable_input_require_grads()
    # print(f"[debugging] torch.cuda.memory, 4: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # print(f"[debugging] torch.cuda.memory, 5: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    
    gc.collect()
    torch.cuda.empty_cache()


    engine = GradDotProdEngine(
        module=model,
        # val_batch_size=len(val_dataset),
        val_batch_size=per_device_train_batch_size//4-1,
        # val_batch_size=8,
        loss_reduction='mean', # or 'sum'
        average_grad=True, # if average_grad=True, then mean is used
        origin_params=None,
    )
    engine.attach(optimizer)
    dot_prod_save_path = os.path.join(output_dir, "grad_dot_products")
    if not os.path.exists(dot_prod_save_path):
        os.makedirs(dot_prod_save_path, exist_ok=True)
    print("GradDotProdEngine initialized successfully.")
    # print(f"[debugging] torch.cuda.memory, 6: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")

    _accelerator = accelerate.Accelerator()
    trainer = _accelerator.prepare(trainer)
    model.train()
    # for name, param in model.named_parameters():
    #     print(f"checking  trainable parameter: {name}")
    #     if param.requires_grad:
    #         print(f"[debugging] Checking again, trainable parameter: {name}, shape: {param.shape}")
    # print(f"[debugging] torch.cuda.memory, 7: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")

    _eval_total = {}
    for _eval_batch in trainer.get_eval_dataloader():
        print(f"{_eval_batch.input_ids.shape=}")
        # tokenized = tokenizer(_eval_batch['text'], truncation=True, padding=True, return_tensors="pt")
        # # _eval_batch[k] = tokenized[k] if k in tokenized else tokenized['input_ids']
        print(_eval_batch.keys())
        for k in _eval_batch:
            if k not in _eval_total.keys():
                _eval_total[k] = _eval_batch[k]
            else:
                print(f"{_eval_total[k].shape=}, {_eval_batch[k].shape=}")
                _eval_total[k] = torch.cat([_eval_total[k], _eval_batch[k]], dim=0)
        break
    # print(f"[debugging] {_eval_total['input_ids'].shape=}, {_eval_total['attention_mask'].shape=}, {_eval_total['labels'].shape=}")
    # print(f"[debugging] torch.cuda.memory, 8: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    # Training loop
    for batch in trainer.get_train_dataloader():
        for k in batch.keys():
            if k in _eval_total:
                batch[k] = torch.cat([batch[k], _eval_total[k]], dim=0)
        # print(f"[debugging] torch.cuda.memory, 9 (in iter): {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
        # print(f"[debugging] in <./pe/llm/ghostsuit.py>, for batch, {batch.keys()=}, {batch.input_ids.shape=}")
        # batch = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}
        # print(f"[debugging] torch.cuda.memory, 9-2 (in iter): {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
        outputs = model(**batch)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        engine.prepare_gradients()
        # Can add any additional gradient operations, such as gradient clipping here. 
        optimizer.step()
        engine.aggregate_and_log()
        engine.clear_gradients()
        # print(f"[debugging] torch.cuda.memory, 10 (in iter): {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")

    engine.save_dot_product_log(save_path=dot_prod_save_path, iter_num=0)
    print("End of training loop")

    # trainer.train()
    # trainer.model.save_pretrained(output_dir) # Save the lora part
    # # trainer.save_model(output_dir)
    # tokenizer.save_pretrained(output_dir)

    # # Save training loss as JSON
    # training_loss_history = []
    # print(f"[debugging], see {trainer.state.log_history=}")
    # train_log = pd.DataFrame(trainer.state.log_history)
    # for log in trainer.state.log_history:
    #     if "train_loss" in log and "step" in log:
    #         training_loss_history.append({
    #             "step": log.get("step", None),
    #             "loss": log.get("train_loss", None)
    #         })
    #     elif 'loss' in log and "step" in log:
    #         training_loss_history.append({
    #             "step": log.get("step", None),
    #             "loss": log.get("loss", None)
    #         })
    # with open(f"{output_dir}/training_loss_history.json", "w") as f:
    #     json.dump(training_loss_history, f, indent=2)
    # logging_plotting(training_loss_history, output_dir)

    # TODO: bug, evaluation of multiple GPUs does not work properly
    # eval_result = trainer.evaluate(train_dataset, metric_key_prefix="train")
    eval_result = None

    model, offload_hook = accelerate.cpu_offload_with_hook(model, execution_device="cuda")
    offload_hook.offload()

    return eval_result, trainer.model, tokenizer



# def ghost_suite_grad_dot(
#     model,
#     tokenizer,
#     train_dataset,
#     val_dataset,
#     output_dir,
#     per_device_train_batch_size=2,
#     num_train_epochs=3,
#     save_steps=500,
#     logging_steps=5,
#     learning_rate=5e-5
# ):
#     print(f"[debugging] {per_device_train_batch_size=}")
#     data_collator = DataCollatorForLanguageModeling(
#         tokenizer=tokenizer, mlm=False
#     )
#     training_args = TrainingArguments(
#         output_dir=output_dir,
#         overwrite_output_dir=True,
#         num_train_epochs=num_train_epochs,
#         per_device_train_batch_size=per_device_train_batch_size,
#         # per_device_train_batch_size=8,
#         save_steps=save_steps,
#         logging_steps=logging_steps,
#         learning_rate=learning_rate,
#         prediction_loss_only=True,
#         report_to=[],
#         remove_unused_columns=False,
#     )
#     print(f"[debugging] torch.cuda.memory, 1: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
#     # Encode the training dataset if not already tokenized
#     if not ("input_ids" in train_dataset[0]):
#         train_dataset = tokenizer(train_dataset['text'], truncation=True, padding=True, return_tensors="pt")
#         # Convert to HuggingFace Dataset if needed
#         train_dataset = Dataset.from_dict(train_dataset)
#     print(f"[debugging] torch.cuda.memory, 2: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")

#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         data_collator=data_collator,
#         train_dataset=train_dataset,
#         eval_dataset=val_dataset,  # Use the same dataset for testing
#     )
#     # for name, param in model.named_parameters():
#     #     if param.requires_grad:
#     #         # param.requires_grad = True
#     #         print(f"[debugging] Trainable parameter: {name}, shape: {param.shape}")
#     print(f"[debugging] torch.cuda.memory, 3: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
#     model.enable_input_require_grads()
#     print(f"[debugging] torch.cuda.memory, 4: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
#     optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#     print(f"[debugging] torch.cuda.memory, 5: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
    
#     gc.collect()
#     torch.cuda.empty_cache()

#     engine = GradDotProdEngine(
#         module=model,
#         # val_batch_size=len(val_dataset),
#         val_batch_size=per_device_train_batch_size,
#         # val_batch_size=8,
#         loss_reduction='mean', # or 'sum'
#         average_grad=True, # if average_grad=True, then mean is used
#         origin_params=None,
#     )
#     engine.attach(optimizer)
#     dot_prod_save_path = os.path.join(output_dir, "grad_dot_products")
#     if not os.path.exists(dot_prod_save_path):
#         os.makedirs(dot_prod_save_path, exist_ok=True)
    print("GradDotProdEngine initialized successfully.")
#     print(f"[debugging] torch.cuda.memory, 6: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")

#     _accelerator = accelerate.Accelerator()
#     trainer = _accelerator.prepare(trainer)
#     model.train()
#     print(f"[debugging] torch.cuda.memory, 7: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")

#     print(f"[debugging] torch.cuda.memory, 8: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
#     # Training loop
#     for batch in trainer.get_train_dataloader():
#         # for k in batch.keys():
#         #     if k in _eval_total:
#         #         batch[k] = torch.cat([batch[k], _eval_total[k][:8]], dim=0)
#         print(f"[debugging] torch.cuda.memory, 9 (in iter): {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
#         print(f"[debugging] in <./pe/llm/ghostsuit.py>, for batch, {batch.keys()=}, {batch.input_ids.shape=}")
#         batch = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}
#         print(f"[debugging] torch.cuda.memory, 9-2 (in iter): {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
#         outputs = model(**batch)
#         loss = outputs.loss
#         optimizer.zero_grad()
#         loss.backward()
#         engine.prepare_gradients()
#         # Can add any additional gradient operations, such as gradient clipping here. 
#         optimizer.step()
#         engine.aggregate_and_log()
#         engine.clear_gradients()
#         print(f"[debugging] torch.cuda.memory, 10 (in iter): {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB, {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")


#     trainer.train()
#     trainer.model.save_pretrained(output_dir) # Save the lora part
#     # trainer.save_model(output_dir)
#     tokenizer.save_pretrained(output_dir)

#     # Save training loss as JSON
#     training_loss_history = []
#     print(f"[debugging], see {trainer.state.log_history=}")
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

#     # TODO: bug, evaluation of multiple GPUs does not work properly
#     # eval_result = trainer.evaluate(train_dataset, metric_key_prefix="train")
#     eval_result = None

#     model, offload_hook = accelerate.cpu_offload_with_hook(model, execution_device="cuda")
#     offload_hook.offload()

#     return eval_result, trainer.model, tokenizer
