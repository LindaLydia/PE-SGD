from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np
import accelerate

def get_per_sample_loss(model, tokenizer, dataset, batch_size=8):
    """
    Compute per-sample loss for a given model and dataset.
    
    Args:
        model: The model to evaluate.
        tokenizer: The tokenizer associated with the model.
        dataset: The dataset to compute losses on.
        batch_size: Batch size for evaluation.
        
    Returns:
        List of per-sample losses.
    """
    model.eval()
    _accelerator = accelerate.Accelerator()
    model = _accelerator.prepare(model)
    # print(F"[debugging] model: {model.device=}, {model.dtype=}")

    dataloader = DataLoader(dataset, batch_size=batch_size)
    # print(f"[debugging] dataloader length: {len(dataloader)=}, {len(dataset)=}")

    per_sample_losses = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing per-sample loss"):
            inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True, max_length=1024)
            input_ids = inputs['input_ids'].to(model.device)
            attention_mask = inputs['attention_mask'].to(model.device)
            labels = input_ids.clone()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask = attention_mask[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(shift_labels.size())

            # # Compute total loss (sum over all tokens, ignoring padding tokens)
            # total_loss = (loss * shift_attention_mask).sum()
            # print(f"[debugging] total_loss: {total_loss.item()}")

            # Compute mean loss over all tokens, ignoring padding tokens)
            loss = (loss * shift_attention_mask).sum(dim=1) / shift_attention_mask.sum(dim=1)

            per_sample_losses.extend(loss.cpu().numpy().tolist())
    # print(f"[debugging] per_sample_losses: {len(per_sample_losses)=}")

    model, offload_hook = accelerate.cpu_offload_with_hook(model, execution_device="cuda")
    offload_hook.offload()

    return torch.tensor(per_sample_losses)