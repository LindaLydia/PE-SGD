import numpy as np
import torch

def image_preprocess(example, processor):
    # Convert PIL.Image to model inputs
    inputs = processor(images=np.asarray(example["image"]), return_tensors="pt")
    # inputs["labels"] = example["label"]
    return {
        "pixel_values": inputs['pixel_values'].squeeze(),  # remove batch dimension
        "labels": torch.tensor(example["labels"]).long().squeeze(),  # ensure labels are of type LongTensor
    }
