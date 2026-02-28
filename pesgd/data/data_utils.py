import copy
import torch
import random
import numpy as np
from datasets import Dataset
from torch.utils.data import DataLoader
from .data import Data
from sklearn.model_selection import train_test_split
from pesgd.logging import execution_logger


def chat_template_tokenize_example(example, prompt_config, tokenizer, max_length=2048):
    # TODO: not supporting classification now
    # sample within sample["text"]
    if "replacement_rules" in prompt_config:
        for replacement_rule in prompt_config["replacement_rules"]:
            constraints = replacement_rule["constraints"]
            replacements = replacement_rule["replacements"]
            satisfied = True
            for key, value in constraints.items():
                if key not in variables or variables[key] != value:
                    satisfied = False
                    break
            if satisfied:
                for key, value in replacements.items():
                    if isinstance(value, list):
                        value = random.choice(value)
                    variables[key] = value
    messages = copy.deepcopy(prompt_config["message_template"])
    if "replacement_rules" in prompt_config:
        for message in messages:
            message["content"] = message["content"].format(**variables)
    
    # add the sample["text"] into it
    messages.append({"role": "assistant", "content": example["text"]})
    # print(f"[debug] in <instruction_addition> {messages=}")

    # Convert messages to a chat-style prompt
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    # print(f"[debug] in <instruction_addition> training and evaluation: {prompt=}")

    # Tokenize the full prompt
    tokenized = tokenizer(prompt, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    input_ids = tokenized["input_ids"].squeeze()
    attention_mask = tokenized["attention_mask"].squeeze()
    # print(f"[debug] in <instruction_addition> {input_ids.shape=}, {attention_mask.shape=}")
    valid_length = (attention_mask == 1.).sum()

    # Mask the prompt portion from loss (labels = -100)
    prompt_only = tokenizer.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
    # print(f"[debug] in <instruction_addition> {prompt_only=}")
    tokenized_prompt_only = tokenizer(prompt_only, return_tensors="pt") # add_special_tokens=False # TODO: test this!!!
    cutoff = tokenized_prompt_only["input_ids"].size(-1)
    # print(f"[debug] in <instruction_addition> {tokenizer(prompt_only, return_tensors='pt')['input_ids'].shape=} {cutoff=}")
    labels = copy.deepcopy(input_ids)
    # print(f"[debug] in <instruction_addition> {labels=}, {labels.shape=}")
    labels[:-valid_length+cutoff] = -100  # only compute loss on assistant's response
    # # labels[:cutoff] = torch.tensor([-100] * cutoff).to(labels.device)  # only compute loss on assistant's response
    # print(f"[debug] in <instruction_addition> {input_ids.shape=}, {cutoff=}, {valid_length=}, {labels.shape=}")
    # print(f"[debug] in <instruction_addition> {input_ids[-1]=}, {labels[-1]=}")

    # print(f"in <instruction_addition.py>, {input_ids=}, {input_ids.shape=}")
    # print(f"in <instruction_addition.py>, {attention_mask=}, {attention_mask.shape=}")
    # print(f"in <instruction_addition.py>, {labels=}, {labels.shape=}")
    # print(f"in <instruction_addition.py>, {attention_mask=}, {attention_mask.shape=}")

    # torch.set_printoptions(threshold=float('inf'))
    # print(f"{input_ids=}")
    # print(f"{attention_mask=}")
    # print(f"{tokenized_prompt_only['input_ids']=}")
    # print(f"{labels=}")
    # print(f"{valid_length=}, {cutoff=}")
    # # assert 1 == 0

    return {
        "input_ids": input_ids.long(),
        "attention_mask": attention_mask.long(),
        "labels": labels.long(),
        # "label": labels.long(),
    }



def prepare_dataset(data, test_size_ratio=0.0):
    """Prepare the private data for the PE algorithm."""
    print(f"{data.data_frame.columns=}")
    if 'PE.TEXT' in data.data_frame.columns:
        if test_size_ratio > 0.0:
            # Split private data into train and dev sets
            train_df, dev_df = train_test_split(data.data_frame, test_size=test_size_ratio, random_state=self._seed, shuffle=True)
            train_data = Dataset.from_pandas(train_df.reset_index(drop=True))
            dev_data = Dataset.from_pandas(dev_df.reset_index(drop=True))
            if not 'text' in train_data.column_names:
                train_data = train_data.add_column("text", train_data['PE.TEXT'])
                dev_data = dev_data.add_column("text", dev_data['PE.TEXT'])
        else:
            train_data = Dataset.from_pandas(data.data_frame.reset_index(drop=True))
            dev_data = None
            if not 'text' in train_data.column_names:
                train_data = train_data.add_column("text", train_data['PE.TEXT'])
                dev_data = None
    elif 'PE.IMAGE' in data.data_frame.columns:
        if test_size_ratio > 0.0:
            # Split private data into train and dev sets
            images, labels, features = self._get_images_and_label_from_data(train_df)
            train_data = Dataset.from_dict({"image": images, "labels": labels}, features=features) if images is not None else None #, features=features
            images, labels, features = self._get_images_and_label_from_data(dev_df)
            dev_data = Dataset.from_dict({"image": images, "labels": labels}, features=features) if images is not None else None #, features=features
            if not 'image' in train_data.column_names:
                train_data = train_data.add_column("image", train_data['PE.IMAGE'])
                dev_data = dev_data.add_column("image", dev_data['PE.IMAGE'])
        else:
            images, labels, features = self._get_images_and_label_from_data(data.data_frame)
            train_data = Dataset.from_dict({"image": images, "labels": labels}, features=features) if images is not None else None #, features=features
            dev_data = None
            if not 'image' in train_data.column_names:
                train_data = train_data.add_column("image", train_data['PE.IMAGE'])
                dev_data = None
    return train_data, dev_data

# def syn_data_preparation(self, syn_data, test_size_ratio=0.0):
#     if 'PE.TEXT' in syn_data.data_frame.columns:
#         if test_size_ratio > 0.0:
#             train_df, dev_df = train_test_split(syn_data, test_size=test_size_ratio, random_state=self._seed, shuffle=True)
#             self._syn_train_data = Dataset.from_pandas(train_df.reset_index(drop=True))
#             self._syn_dev_data = Dataset.from_pandas(dev_df.reset_index(drop=True))
#             # self._syn_eval_data = Dataset.from_pandas(syn_data.reset_index(drop=True))
#             if not 'text' in self._syn_train_data.column_names:
#                 self._syn_train_data = self._syn_train_data.add_column("text", self._syn_train_data['PE.TEXT'])
#                 self._syn_dev_data = self._syn_dev_data.add_column("text", self._syn_dev_data['PE.TEXT'])
#                 # self._syn_eval_data = self._syn_eval_data.add_column("text", self._syn_eval_data['PE.TEXT'])
#         else:
#             self._syn_train_data = Dataset.from_pandas(syn_data.data_frame.reset_index(drop=True))
#             self._syn_dev_data = None
#             # self._syn_eval_data = Dataset.from_pandas(syn_data.data_frame.reset_index(drop=True))
#             self._syn_eval_data = None
#             if not 'text' in self._syn_train_data.column_names:
#                 self._syn_train_data = self._syn_train_data.add_column("text", self._syn_train_data['PE.TEXT'])
#                 self._syn_dev_data = None
#                 # self._syn_eval_data = self._syn_eval_data.add_column("text", self._syn_eval_data['PE.TEXT'])
#                 self._syn_eavl_data = None
#     elif 'PE.IMAGE' in syn_data.data_frame.columns:
#         if test_size_ratio > 0.0:
#             train_df, dev_df = train_test_split(syn_data.data_frame, test_size=test_size_ratio, random_state=self._seed, shuffle=True)
#             images, labels, features = self._get_images_and_label_from_data(train_df)
#             self._syn_train_data = Dataset.from_dict({"image": images, "labels": labels}, features=features) if images is not None else None #, features=features
#             images, labels, features = self._get_images_and_label_from_data(dev_df)
#             self._syn_dev_data = Dataset.from_dict({"image": images, "labels": labels}, features=features) if images is not None else None #, features=features
#             if not 'image' in self._syn_train_data.column_names:
#                 self._syn_train_data = self._syn_train_data.add_column("image", self._syn_train_data['PE.IMAGE'])
#                 self._syn_dev_data = self._syn_dev_data.add_column("image", self._syn_dev_data['PE.IMAGE'])
#                 # self._syn_eval_data = self._syn_eval_data.add_column("image", self._syn_eval_data['PE.IMAGE'])
#             # if not 'text' in self._syn_train_data.column_names:
#             #     self._syn_train_data = self._syn_train_data.add_column("text", self._syn_train_data['image'])
#             #     self._syn_dev_data = self._syn_dev_data.add_column("text", self._syn_dev_data['image'])
#             #     # self._syn_eval_data = self._syn_eval_data.add_column("text", self._syn_eval_data['image'])
#         else:
#             images, labels, features = self._get_images_and_label_from_data(syn_data.data_frame)
#             self._syn_train_data = Dataset.from_dict({"image": images, "labels": labels}, features=features) if images is not None else None #, features=features
#             self._syn_dev_data = None
#             # self._syn_eval_data = Dataset.from_pandas(syn_data.data_frame.reset_index(drop=True))
#             if not 'image' in self._syn_train_data.column_names:
#                 self._syn_train_data = self._syn_train_data.add_column("image", self._syn_train_data['PE.IMAGE'])
#                 self._syn_dev_data = None
#                 # self._syn_eval_data = self._syn_eval_data.add_column("image", self._syn_eval_data['PE.IMAGE'])
#                 self._syn_eavl_data = None
#             # if not 'text' in self._syn_train_data.column_names:
#             #     self._syn_train_data = self._syn_train_data.add_column("text", self._syn_train_data['image'])
#             #     self._syn_dev_data = None
#             #     # self._syn_eval_data = self._syn_eval_data.add_column("text", self._syn_eval_data['image'])
#             #     self._syn_eavl_data = None



def prepared_dataloader(data, processor, split_ratio=0.0, task_type='text', batch_size=8, shuffle=True, add_instruction=False, instruction=None):
    dataset, _ = prepare_dataset(data)
    if task_type == 'image':
        image_preprocess = processor
        # Encode the training dataset if not already tokenized
        if not ("pixel_values" in dataset[0]):
            dataset = dataset.map(image_preprocess, fn_kwargs={"processor": tokenizer})
            dataset.set_format(type='torch', columns=['pixel_values', 'labels'])
    else:
        tokenizer = processor
        # Encode the training dataset if not already tokenized
        if not ("input_ids" in dataset[0]):
            if not add_instruction:
                # print(f"[debugging] dataset={dataset}")
                dataset = tokenizer(dataset['text'], truncation=True, padding=True, return_tensors="pt")
                dataset["labels"] = dataset["input_ids"].clone()
                dataset["labels"][dataset["attention_mask"] == 0] = -100
                # Convert to HuggingFace Dataset
                dataset = Dataset.from_dict(dataset)
                dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
            else:
                assert instruction != None, f'[ERROR] When specifying add_instruction=True, instruction should be provided.'
                dataset = dataset.map(
                                    chat_template_tokenize_example, 
                                    fn_kwargs={"prompt_config": instruction, "tokenizer": tokenizer, "max_length": 2048}, 
                                    remove_columns=dataset.column_names
                                ) #, remove_columns=dataset["train"].column_names
                dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
                # print(f"[debugging] in <./pesgd/data/data_utils.py> <prepare_dataloader>, {dataset.column_names=}")

    # print(f"[debugging] in <./pesgd/data/data_utils.py> <prepare_dataloader>, {dataset[0]=}")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size, shuffle=shuffle # , collate_fn=data_collator  # collates dicts into tensors
    )
    # print(f"[debugging] in <./pesgd/data/data_utils.py> <prepare_dataloader>, {dataloader.dataset[0]=}")

    return dataset, dataloader


def get_num_samples_per_label_id(data_metadata, num_samples, fraction_per_label_id):
    """Get the number of samples per label id given the total number of samples

    :param num_samples: The total number of samples
    :type num_samples: int
    :param fraction_per_label_id: The fraction of samples for each label id. The fraction does not have to be
        normalized. When it is None, the fraction is assumed to be the same as the fraction of label ids in the
        private data. Defaults to None
    :type fraction_per_label_id: list[float], optional
    :raises ValueError: If the length of fraction_per_label_id is not the same as the number of labels
    :raises ValueError: If the number of samples is so small that the number of samples for some label ids is zero
    :return: The number of samples per label id
    :rtype: np.ndarray
    """
    if fraction_per_label_id is None:
        execution_logger.warning(
            "fraction_per_label_id is not provided."
            # "Assuming the fraction of label ids in private data is public information."
            "Assuming the fraction of label ids in private data is equally distributed."
        )
        # fraction_per_label_id = self._priv_data.data_frame[LABEL_ID_COLUMN_NAME].value_counts().to_dict()
        # fraction_per_label_id = [
        #     0 if i not in fraction_per_label_id else fraction_per_label_id[i]
        #     for i in range(len(data_metadata.label_info))
        # ]
        fraction_per_label_id = [1.0/len(data_metadata.label_info)] * len(data_metadata.label_info)
    if len(fraction_per_label_id) != len(data_metadata.label_info):
        raise ValueError("fraction_per_label_id should have the same length as the number of labels.")
    fraction_per_label_id = np.array(fraction_per_label_id)
    fraction_per_label_id = fraction_per_label_id / np.sum(fraction_per_label_id)

    target_num_samples_per_label_id = fraction_per_label_id * num_samples
    num_samples_per_label_id = np.floor(target_num_samples_per_label_id).astype(int)
    num_samples_left = num_samples - np.sum(num_samples_per_label_id)
    ids = np.argsort(target_num_samples_per_label_id - num_samples_per_label_id)[::-1]
    num_samples_per_label_id[ids[:num_samples_left]] += 1
    assert np.sum(num_samples_per_label_id) == num_samples
    if np.any(num_samples_per_label_id == 0):
        raise ValueError("num_samples is so small that the number of samples for some label ids is zero.")
    return num_samples_per_label_id


def load_checkpoint(checkpoint_path):
    """Load a checkpoint.

    :param checkpoint_path: The path to the checkpoint
    :type checkpoint_path: str
    :return: The synthetic data
    :rtype: :py:class:`pe.data.Data` or None
    """
    syn_data = Data()
    if not syn_data.load_checkpoint(checkpoint_path):
        return None
    return syn_data


def log_metrics(syn_data, callbacks=None, loggers=None):
    """Log metrics.

    :param syn_data: The synthetic data
    :type syn_data: :py:class:`pe.data.Data`
    """
    if not callbacks:
        return
    metric_items = []
    for callback in callbacks:
        metric_items.extend(callback(syn_data) or [])
    for logger in loggers:
        logger.log(iteration=syn_data.metadata.iteration, metric_items=metric_items)
    for metric_item in metric_items:
        metric_item.clean_up()


def get_initial_none_priv_data(data_metadata, checkpoint_path=None, init_data_file=None, num_samples_schedule=[], fraction_per_label_id=None, population=None, callbacks=None, loggers=None):
    """Generate or load the initial synthetic data."""
    # Generate or load initial data.
    if checkpoint_path is not None and (syn_data := load_checkpoint(checkpoint_path)):
        execution_logger.info(
            f"Loaded checkpoint from {checkpoint_path}, iteration={syn_data.metadata.iteration}"
        )
    elif init_data_file is not None and (syn_data := load_checkpoint(init_data_file)):
        execution_logger.info(f"PE initial data loaded from already generated data with {len(syn_data.data_frame)} samples [finished].")
    else:
        num_samples_per_label_id = get_num_samples_per_label_id(
            data_metadata=data_metadata,
            num_samples=num_samples_schedule[0],
            fraction_per_label_id=fraction_per_label_id,
        )
        syn_data_list = []
        for label_id, label_info in enumerate(data_metadata.label_info):
            syn_data = population.initial(
                label_info=label_info,
                num_samples=num_samples_per_label_id[label_id],
            )
            syn_data.set_label_id(label_id)
            syn_data_list.append(syn_data)
        syn_data = Data.concat(syn_data_list, metadata=data_metadata)
        syn_data.data_frame.reset_index(drop=True, inplace=True)
        syn_data.metadata.iteration = 0
        log_metrics(syn_data, callbacks=callbacks, loggers=loggers)
        execution_logger.info(f"PE initial data generated with {len(syn_data.data_frame)} samples [finished].")
        if init_data_file is not None:
            syn_data.save_checkpoint(init_data_file)
            print(f"save PE initial data to {init_data_file}")

    return syn_data

