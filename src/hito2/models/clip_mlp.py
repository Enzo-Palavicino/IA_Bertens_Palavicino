"""Simple configurable MLP for CLIP embedding classification."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class ClipMLPClassifier(nn.Module):
    """A lightweight MLP head on top of frozen CLIP image embeddings."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_sizes: Sequence[int] = (512, 256),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        hidden_sizes = tuple(int(size) for size in hidden_sizes)
        if any(size <= 0 for size in hidden_sizes):
            raise ValueError(f"All hidden sizes must be positive integers. Received: {hidden_sizes}")
        if input_dim <= 0:
            raise ValueError(f"`input_dim` must be positive. Received: {input_dim}")
        if num_classes <= 1:
            raise ValueError(f"`num_classes` must be greater than 1. Received: {num_classes}")

        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers) if layers else nn.Identity()
        self.classifier = nn.Linear(previous_dim, num_classes)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return multiclass logits for one batch of CLIP embeddings."""
        hidden = self.feature_extractor(embeddings)
        return self.classifier(hidden)
