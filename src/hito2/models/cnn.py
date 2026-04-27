"""Configurable CNN baseline for Hito 2 multiclass classification."""

from __future__ import annotations

from typing import Sequence

from torch import nn


class SimpleCNNClassifier(nn.Module):
    """Compact CNN suitable for local multiclass training on resized query images."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        conv_channels: Sequence[int] = (32, 64, 128),
        classifier_hidden_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        conv_channels = tuple(int(channel) for channel in conv_channels)
        if num_classes <= 1:
            raise ValueError(f"`num_classes` must be greater than 1. Received: {num_classes}")
        if in_channels <= 0:
            raise ValueError(f"`in_channels` must be positive. Received: {in_channels}")
        if any(channel <= 0 for channel in conv_channels):
            raise ValueError(f"All `conv_channels` must be positive. Received: {conv_channels}")
        if classifier_hidden_dim <= 0:
            raise ValueError(
                f"`classifier_hidden_dim` must be positive. Received: {classifier_hidden_dim}"
            )

        layers: list[nn.Module] = []
        previous_channels = in_channels
        for out_channels in conv_channels:
            layers.extend(
                [
                    nn.Conv2d(previous_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ]
            )
            previous_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(previous_channels, classifier_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    def forward(self, images):
        """Return multiclass logits for a batch of images."""
        features = self.features(images)
        pooled = self.pool(features)
        return self.classifier(pooled)
