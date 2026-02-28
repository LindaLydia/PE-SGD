import torch
from transformers import Trainer


def per_sample_loss_function_image(labels, logits, reduction='mean'): #

    shift_labels = labels
    shift_logits = logits
    print(f"in <./pe/llm/fine_tune/trainer_image.py> {shift_logits=}, {shift_labels=}")
    print(f"in <./pe/llm/fine_tune/trainer_image.py> {shift_logits.shape=}, {shift_labels.shape=}")
    
    # Calculate per-token loss
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
    loss_per_sample = loss_fct(
        # torch.nn.functional.softmax(shift_logits.view(-1, shift_logits.size(-1)), dim=-1).to(dtype=torch.float32),  # [batch, vocab]
        shift_logits.view(-1, shift_logits.size(-1)).to(dtype=torch.float32),  # [batch, vocab]
        shift_labels.view(-1)                                                  # [batch]
    )
    loss_per_sample = loss_per_sample.view(-1)  # Flatten to [batch_size]
    print(f"in <./pe/llm/fine_tune/trainer_image.py> {shift_logits.view(-1, shift_logits.size(-1)).shape=}, {shift_labels.shape=}")
    print(f"in <./pe/llm/fine_tune/trainer_image.py> {loss_per_sample=}, {loss_per_sample.shape=}")
    
    if reduction == 'mean':
        return loss_per_sample.mean()
    elif reduction == 'sum':
        return loss_per_sample.sum()
    else:
        # Return per-sample loss
        return loss_per_sample


class PerSampleLossTrainerImage(Trainer):
    def __init__(self, *args, loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = per_sample_loss_function_image if loss_fn is None else loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        print(f"before {labels.shape=}, {logits.shape=}")
        logits = outputs.logits.view(labels.size(0), -1)
        print(f"after {labels.shape=}, {logits.shape=}")
        # assert 0 == 1

        # Flatten the predictions and labels for loss calc
        loss = self.loss_fn( # will return a tensor of shape [batch_size], so use .mean() to get the average loss if needed
            labels, logits,
        ) 

        return (loss, outputs) if return_outputs else loss

    def compute_loss_func(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        print(f"before {labels.shape=}, {logits.shape=}")
        logits = outputs.logits.view(labels.size(0), -1)
        print(f"after {labels.shape=}, {logits.shape=}")
        # assert 0 == 1

        # Flatten the predictions and labels for loss calc
        loss = self.loss_fn( # will return a tensor of shape [batch_size], so use .mean() to get the average loss if needed
            labels, logits,
        ) 

        return (loss, outputs) if return_outputs else loss