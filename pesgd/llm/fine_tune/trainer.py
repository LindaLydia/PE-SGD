import torch
from transformers import Trainer


def per_sample_loss_function(labels, logits, shift_label=True, reduction='mean'): #
    if shift_label:
        # print(f"in <./pesgd/llm/fine_tune/trainer.py> <per_sample_loss_function>, {logits=}, {labels=}, {logits.shape=}, {labels.shape=}")
        shift_labels = torch.nn.functional.pad(labels, (0, 1), value=-100)
        shift_labels = shift_labels[..., 1:].contiguous()
        shift_logits = logits
        # print(f"in <./pesgd/llm/fine_tune/trainer.py> <per_sample_loss_function>, {shift_logits=}, {shift_labels=}, {shift_logits.shape=}, {shift_labels.shape=}")
    else:
        shift_labels = labels
        shift_logits = logits
    # print(f"{shift_logits.shape=}, {shift_labels.shape=}")
    
    # Calculate per-token loss
    loss_fct = torch.nn.CrossEntropyLoss(reduce=False, ignore_index=-100)
    loss_per_token = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)).to(dtype=torch.float32),  # [batch*seq, vocab]
        shift_labels.view(-1)                                                  # [batch*seq]
    )
    loss_per_token = loss_per_token.view(shift_labels.size())  # [batch, seq_len]

    # Mask ignored tokens (e.g., padding with -100)
    mask = shift_labels != -100
    loss_per_token = loss_per_token * mask
    # Average over tokens per sample → per-sample loss
    loss_per_sample = loss_per_token.sum(dim=1) / mask.sum(dim=1)
    loss_per_sample = loss_per_sample.view(-1)  # Flatten to [batch_size]
    # print(f"{loss_per_sample=}, {loss_per_token.sum()/mask.sum()}")
    
    if reduction == 'mean':
        return loss_per_sample.mean()
    elif reduction == 'sum':
        return loss_per_sample.sum()
    else:
        # Return per-sample loss
        return loss_per_sample


class PerSampleLossTrainer(Trainer):
    def __init__(self, *args, loss_fn=None, shift_loss=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = per_sample_loss_function if loss_fn is None else loss_fn
        self.shift_loss = shift_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # print(f"before {labels.shape=}, {logits.shape=}")
        logits = outputs.logits.view(labels.size(0), labels.size(1), -1)
        # print(f"after {labels.shape=}, {logits.shape=}")
        # assert 0 == 1

        # Flatten the predictions and labels for loss calc
        loss = self.loss_fn( # will return a tensor of shape [batch_size], so use .mean() to get the average loss if needed
            labels, logits, shift_label=self.shift_loss,
        ) 

        return (loss, outputs) if return_outputs else loss

    def compute_loss_func(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # print(f"before {labels.shape=}, {logits.shape=}")
        logits = outputs.logits.view(labels.size(0), labels.size(1), -1)
        # print(f"after {labels.shape=}, {logits.shape=}")
        # assert 0 == 1

        # Flatten the predictions and labels for loss calc
        loss = self.loss_fn( # will return a tensor of shape [batch_size], so use .mean() to get the average loss if needed
            labels, logits, shift_label=self.shift_loss,
        ) 

        return (loss, outputs) if return_outputs else loss
