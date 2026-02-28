import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import accelerate
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset
import copy
import gc

from pe.logging import execution_logger
from .instruction_addition import chat_template_tokenize_example
from .trainer import per_sample_loss_function
from .trainer_image import per_sample_loss_function_image
from .image_dataset_processing import image_preprocess


def evaluate_model_on_private_data(model, tokenizer, dataloader):
    
    _accelerator = accelerate.Accelerator()
    model, tokenizer, dataloader = _accelerator.prepare(model, tokenizer, dataloader)
    model.eval()
    device = next(model.parameters()).device
    # print(f"[debugging] in <pesgd.llm.fine_tune.fine_tuned_model_eval> func <evaluate_model_on_private_data>, {device=}")

    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0

    # if not 'ImageProcessor' in tokenizer.__class__.__name__:
    #     original_padding_side = tokenizer.padding_side
    #     tokenizer.padding_side = "left"

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating on private data"):
            if 'ImageProcessor' in tokenizer.__class__.__name__:
                # pixel_values = batch["pixel_values"].to(device)
                # labels = batch["labels"].to(device)
                batch = {key: val.to(next(model.parameters()).device) for key, val in batch.items()}
                inputs = {'pixel_values':    batch['pixel_values'],
                        # 'attention_mask': batch['attention_mask'],
                        # 'token_type_ids': batch[2],
                        'labels':         batch['labels']}
                pixel_values = inputs["pixel_values"]
                labels = inputs['labels']
                outputs = model(pixel_values=pixel_values, labels=labels)
                logits = outputs.logits
                loss = per_sample_loss_function_image(labels, logits, reduction='none').sum()
                total_loss += loss.item()

                # Token prediction accuracy
                preds = torch.argmax(logits, dim=-1)
                correct = (preds == labels)

                correct_tokens += correct.sum().item()
                total_tokens += len(labels)

                del pixel_values, labels, logits, loss, preds, correct, mask
            
            else:  # for text with instruction
                # torch.set_printoptions(threshold=10000)
                batch = {key: val.to(next(model.parameters()).device) for key, val in batch.items()}
                inputs = {'input_ids':    batch['input_ids'],
                        'attention_mask': batch['attention_mask'],
                        # 'token_type_ids': batch[2],
                        'labels':         batch['labels']}
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                labels = inputs['labels']

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits
                # print(f"[debugging] in <pesgd.llm.fine_tune.fine_tuned_model_eval> func <evaluate_model_on_private_data>, {input_ids=}")
                # print(f"[debugging] in <pesgd.llm.fine_tune.fine_tuned_model_eval> func <evaluate_model_on_private_data>, {attention_mask=}")
                # print(f"[debugging] in <pesgd.llm.fine_tune.fine_tuned_model_eval> func <evaluate_model_on_private_data>, {labels=}")
                # for _item in labels[0]:
                #     print(f"{_item=}")
                # print(f"[debugging] in <pesgd.llm.fine_tune.fine_tuned_model_eval> func <evaluate_model_on_private_data>, {logits=}")
                loss = outputs.loss
                loss = per_sample_loss_function(labels, logits, shift_label=True, reduction='none').sum()
                total_loss += loss.item()

                # Token prediction accuracy
                preds = torch.argmax(logits, dim=-1)
                # print(f"[debugging] in <pesgd.llm.fine_tune.fine_tuned_model_eval> func <evaluate_model_on_private_data>, {preds=}")
                # print(f"[debugging] in <pesgd.llm.fine_tune.fine_tuned_model_eval> func <evaluate_model_on_private_data>, {labels=}")
                # print(f"[debugging] in <pesgd.llm.fine_tune.fine_tuned_model_eval> func <evaluate_model_on_private_data>, {loss=}，{outputs.loss=}")
                preds = preds[:,:-1]
                labels = labels[:,1:]
                mask = labels != -100
                correct = (preds == labels) & mask

                correct_tokens += correct.sum().item()
                total_tokens += mask.sum().item()

                del input_ids, attention_mask, labels, logits, loss, preds, correct, mask
                # torch.set_printoptions(threshold=30)
            torch.cuda.empty_cache()
            gc.collect()

    model, offload_hook = accelerate.cpu_offload_with_hook(model, execution_device="cuda")
    offload_hook.offload()

    # if not 'ImageProcessor' in tokenizer.__class__.__name__:
    #     tokenizer.padding_side = original_padding_side

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = correct_tokens / total_tokens
    return avg_loss, accuracy



# def evaluate_model_on_private_data(model, tokenizer, dataset, batch_size=8, add_instruction=False, instruction=None):
#     if add_instruction == False:
#         instruction = None

#     if 'ImageProcessor' in tokenizer.__class__.__name__:
#         # Encode the training dataset if not already tokenized
#         if not ("pixel_values" in dataset[0]):
#             dataset = dataset.map(image_preprocess, fn_kwargs={"processor": tokenizer})
#             dataset.set_format(type='torch', columns=['pixel_values', 'labels'])
#         dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=False) #, collate_fn=data_collator
#     else:
#         if add_instruction == True:
#             assert instruction != None, f'[ERROR] When specifying add_instruction=True, instruction should be provided.'
#             if not "labels" in dataset.column_names:
#                print(f"mapping")
#                 dataset = dataset.map(
#                                 chat_template_tokenize_example, 
#                                 fn_kwargs={"prompt_config": instruction, "tokenizer": tokenizer, "max_length": 2048}, 
#                                 remove_columns=dataset.column_names
#                             ) #, remove_columns=train_dataset["train"].column_names
#                 dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
#             # data_collator = DataCollatorForLanguageModeling(
#             #     tokenizer=tokenizer, mlm=False
#             # )
#             # torch.set_printoptions(threshold=float('inf'))
            # print(f"{dataset['labels'][0]}")
#             dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=False) #, collate_fn=data_collator
#         else:
#             if isinstance(dataset, Dataset):
#                 need_tokenization = ('text' in dataset.column_names) and (not 'input_ids' in dataset.column_names)
#             else:
#                 need_tokenization = ('text' in dataset.keys()) and (not 'input_ids' in dataset.keys())
#             if need_tokenization:
#                 dataset = tokenizer(dataset['text'], truncation=True, padding="max_length", max_length=1024, return_tensors="pt")
#                 # dataset = tokenizer(tokenizer.tokenize(dataset['text']), truncation=True, padding="max_length", max_length=1024, return_tensors="pt")
#                 dataset['labels'] = copy.deepcopy(dataset['input_ids'])
#                 dataset = Dataset.from_dict(dataset)
#             else:
#                 if isinstance(dataset, Dataset):
#                     if (not 'labels' in dataset.column_names):
#                         dataset = dataset.add_column('labels', copy.deepcopy(dataset['input_ids']))
#                 else:
#                     if (not 'labels' in dataset.keys()):
#                         dataset['labels'] = copy.deepcopy(dataset['input_ids'])
#                         dataset = Dataset.from_dict(dataset)
#             # data_collator = DataCollatorForLanguageModeling(
#             #     tokenizer=tokenizer, mlm=False
#             # )
#             dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
#             dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False) #, collate_fn=data_collator)
    
#     _accelerator = accelerate.Accelerator()
#     model, tokenizer, dataloader = _accelerator.prepare(model, tokenizer, dataloader)
#     model.eval()
#     device = next(model.parameters()).device
#     print(f"[debugging] in <./pe/llm/fine_tune/fine_tuned_model_eval.py> {device=}")

#     total_loss = 0.0
#     total_tokens = 0
#     correct_tokens = 0

#     with torch.no_grad():
#         for batch in tqdm(dataloader, desc="Evaluating on private data"):
            
#             if 'ImageProcessor' in tokenizer.__class__.__name__:
#                 pixel_values = batch["pixel_values"].to(device)
#                 labels = batch["labels"].to(device)
#                 outputs = model(pixel_values=pixel_values, labels=labels)
#                 logits = outputs.logits
#                 loss = per_sample_loss_function_image(labels, logits, reduction='none').sum()
#                 total_loss += loss.item()

#                 # Token prediction accuracy
#                 preds = torch.argmax(logits, dim=-1)
#                 correct = (preds == labels)

#                 correct_tokens += correct.sum().item()
#                 total_tokens += len(labels)
            
#             elif not instruction: # for text without instruction
                # print(f"gpt2, no-instruction should be used.")
#                 if 'text' in batch:
#                     inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True)
#                     input_ids = inputs["input_ids"].to(device)
#                     attention_mask = inputs["attention_mask"].to(device)
#                     labels = input_ids.clone()
#                 else:
#                     input_ids = batch["input_ids"].to(device)
#                     attention_mask = batch["attention_mask"].to(device)
#                     labels = input_ids.clone()
                # print(f"{input_ids=}, {input_ids.shape=}")
                # print(f"{attention_mask=}, {attention_mask.shape=}")
                # print(f"{labels=}, {labels.shape=}")
#                 outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#                 logits = outputs.logits
#                 loss = per_sample_loss_function(labels, logits, reduction='none').sum()

#                 total_loss += loss.item()
                
#                 # Compute token-level accuracy
#                 predictions = logits.argmax(dim=-1)
#                 # predictions = predictions[:,:-1]
#                 # labels = labels[:,1:]
#                 # mask = attention_mask[:,1:].bool()
#                 mask = attention_mask.bool()
#                 # mask = labels != int(tokenizer.eos_token_id)
#                 correct = (predictions == labels) & mask
#                 correct_tokens += correct.sum().item()
#                 total_tokens += mask.sum().item()
#             else:  # for text with instruction
#                 if type(batch["input_ids"]) == torch.Tensor:
#                     input_ids = batch["input_ids"].to(device)
#                     attention_mask = batch["attention_mask"].to(device)
#                     labels = batch["labels"].to(device)
#                 else:
#                     input_ids = torch.tensor(batch["input_ids"]).to(device)
#                     attention_mask = torch.tensor(batch["attention_mask"]).to(device)
#                     labels = torch.tensor(batch["labels"]).to(device)

#                 outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#                 logits = outputs.logits
#                 loss = outputs.loss
#                 loss = per_sample_loss_function(labels, logits, reduction='none').sum()
#                 total_loss += loss.item()

#                 # Token prediction accuracy
#                 preds = torch.argmax(logits, dim=-1)
#                 preds = preds[:,:-1]
#                 labels = labels[:,1:]
#                 mask = labels != -100
#                 correct = (preds == labels) & mask

#                 correct_tokens += correct.sum().item()
#                 total_tokens += mask.sum().item()

#     model, offload_hook = accelerate.cpu_offload_with_hook(model, execution_device="cuda")
#     offload_hook.offload()

#     if not instruction:
#         # avg_loss = total_loss / total_tokens
#         avg_loss = total_loss / len(dataset)
#         accuracy = correct_tokens / total_tokens
#     else:
#         avg_loss = total_loss / len(dataset)
#         accuracy = correct_tokens / total_tokens
#     return avg_loss, accuracy


def evaluate_model_by_sample(model, tokenizer, dataset, batch_size=8, add_instruction=False, instruction=None):
    if add_instruction == False:
        instruction = None

    if add_instruction == True:
        assert instruction != None, f'[ERROR] When specifying add_instruction=True, instruction should be provided.'
        if not "labels" in dataset.column_names:
            print(f"mapping")
            dataset = dataset.map(
                            chat_template_tokenize_example, 
                            fn_kwargs={"prompt_config": instruction, "tokenizer": tokenizer, "max_length": 2048}, 
                            remove_columns=dataset.column_names
                        ) #, remove_columns=train_dataset["train"].column_names
            dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        # data_collator = DataCollatorForLanguageModeling(
        #     tokenizer=tokenizer, mlm=False
        # )
        # torch.set_printoptions(threshold=float('inf'))
        print(f"{dataset['labels'][0]}")
        dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=False) #, collate_fn=data_collator
    else:
        if isinstance(dataset, Dataset):
            need_tokenization = ('text' in dataset.column_names) and (not 'input_ids' in dataset.column_names)
        else:
            need_tokenization = ('text' in dataset.keys()) and (not 'input_ids' in dataset.keys())
        if need_tokenization:
            dataset = tokenizer(dataset['text'], truncation=True, padding=True, return_tensors="pt")
            dataset = Dataset.from_dict(dataset)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    
    _accelerator = accelerate.Accelerator()
    model, tokenizer, dataloader = _accelerator.prepare(model, tokenizer, dataloader)
    model.eval()
    device = next(model.parameters()).device
    # print(f"[debugging] in <./pe/llm/fine_tune/fine_tuned_model_eval.py> {device=}")

    loss_list = []
    acc_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating on private data"):
            # Assume 'text' column exists
            if not instruction:
                print(f"gpt2, no-instruction should be used.")
                if 'text' in batch:
                    inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True)
                    input_ids = inputs["input_ids"].to(device)
                    attention_mask = inputs["attention_mask"].to(device)
                    labels = input_ids.clone()
                else:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = input_ids.clone()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits
                loss = per_sample_loss_function(labels, logits, reduction='none')

                # Compute token-level accuracy
                predictions = logits.argmax(dim=-1)
                mask = attention_mask.bool()
                predictions = predictions[:,:-1]
                labels = labels[:,1:]
                mask = mask[:,1:]
                correct = (predictions == labels) & mask
                acc = correct.sum(dim=-1) / mask.sum(dim=-1)

                loss_list.extend(loss.cpu().tolist())
                acc_list.extend(acc.cpu().tolist())
            else:
                if type(batch["input_ids"]) == torch.Tensor:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                else:
                    input_ids = torch.tensor(batch["input_ids"]).to(device)
                    attention_mask = torch.tensor(batch["attention_mask"]).to(device)
                    labels = torch.tensor(batch["labels"]).to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits
                loss = outputs.loss
                loss = per_sample_loss_function(labels, logits, reduction='none')
                
                # Compute token-level accuracy
                preds = torch.argmax(logits, dim=-1)
                preds = preds[:,:-1]
                labels = labels[:,1:]
                mask = labels != -100
                correct = (preds == labels) & mask
                print(f"{correct.shape=}, {mask.shape=}, {correct.sum(dim=-1)=}, {mask.sum(dim=-1)=}")
                acc = correct.sum(dim=-1) / mask.sum(dim=-1)
                print(f"{acc=}, {acc.cpu().bfloat16()=}")

                loss_list.extend(loss.cpu().tolist())
                acc_list.extend(acc.cpu().tolist())

    model, offload_hook = accelerate.cpu_offload_with_hook(model, execution_device="cuda")
    offload_hook.offload()

    return loss_list, acc_list



def next_token_accuracy(eval_preds):
    """
    Compute the next token accuracy.
    """
    # print(f"[debugging] in <./pe/llm/fine_tune/fine_tuned_model_eval.py> <next_token_accuracy> {eval_preds=}")
    predictions, labels = eval_preds
    predictions = torch.argmax(predictions, axis=-1)

    preds = predictions[:,:-1]
    labels = labels[:,1:]
    
    # Remove padding tokens from labels
    mask = labels != -100
    correct_predictions = ((preds == labels) & mask).sum().item()
    # correct_predictions = (predictions[mask] == labels[mask]).sum()
    total_predictions = mask.sum().item()
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    return {"accuracy": accuracy}
