from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from torchvision.transforms.functional import pil_to_tensor


class MyCustomModel(nn.Module):
    def __init__(self, n_classes: int = 200, resolution: int = 64, intermidiate_dimensions: List[int] = [128, 256]):
        super().__init__()
        self.n_classes = n_classes
        self.resolution = resolution
        self.intermidiate_dimensions = intermidiate_dimensions

        # Build model
        layer_dims = [resolution * resolution] + intermidiate_dimensions + [n_classes]
        layers = []
        for input_dim, output_dim in zip(layer_dims[:-1], layer_dims[1:]):
            layers.append(nn.Linear(in_features=input_dim, out_features=output_dim))
            if output_dim != n_classes:
                layers.append(nn.ReLU())

        self.model = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x


@dataclass
class MyCustomCollator:
    resolution: int = 64

    def __call__(self, samples):
        # Convert RGB --> Gray scale & Resize
        inputs = [sample["image"].convert("L").resize((self.resolution, self.resolution)) for sample in samples]
        # Convert PIL image to torch.tensor
        inputs = [pil_to_tensor(sample).to(torch.float32) for sample in inputs]
        # Reshape properly before feeding the tensor into the model
        # TIP: We use `torch.stack` to create the batch dimension!
        inputs = torch.stack([sample.flatten() for sample in inputs])

        # Convert labels (int) to torch.tensor
        labels = torch.tensor([torch.tensor(sample["label"]) for sample in samples])

        return inputs, labels


@dataclass
class RGBCollator:
    resolution: int

    def __call__(self, samples):
        # Resize
        inputs = [sample["image"].resize((self.resolution, self.resolution)) for sample in samples]
        # Convert PIL image to torch.tensor
        inputs = [pil_to_tensor(sample).to(torch.float32) for sample in inputs]
        # Reshape properly before feeding the tensor into the model
        # TIP: We use `torch.stack` to create the batch dimension!
        inputs = torch.stack([sample for sample in inputs])

        # Convert labels (int) to torch.tensor
        labels = torch.tensor([torch.tensor(sample["label"]) for sample in samples])

        return inputs, labels
