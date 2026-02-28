import copy
import torch


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

    print(f"in <instruction_addition.py>, {input_ids=}, {input_ids.shape=}")
    print(f"in <instruction_addition.py>, {attention_mask=}, {attention_mask.shape=}")
    print(f"in <instruction_addition.py>, {labels=}, {labels.shape=}")
    print(f"in <instruction_addition.py>, {attention_mask=}, {attention_mask.shape=}")

    # torch.set_printoptions(threshold=float('inf'))
    print(f"{input_ids=}")
    print(f"{attention_mask=}")
    print(f"{tokenized_prompt_only['input_ids']=}")
    print(f"{labels=}")
    print(f"{valid_length=}, {cutoff=}")
    # assert 1 == 0

    return {
        "input_ids": input_ids.long(),
        "attention_mask": attention_mask.long(),
        "labels": labels.long(),
        # "label": labels.long(),
    }

