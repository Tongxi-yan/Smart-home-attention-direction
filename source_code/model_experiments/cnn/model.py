"""1D-CNN baseline from the final experiment set."""

from __future__ import annotations

import torch
from torch import nn


class CNN1DClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int = 3,
        channels: tuple[int, int] = (128, 256),
        kernel_size: int = 5,
        dropout: float = 0.3,
        classifier_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        first, second = channels
        padding = kernel_size // 2
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_size, first, kernel_size, padding=padding),
            nn.BatchNorm1d(first),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(first, second, kernel_size, padding=padding),
            nn.BatchNorm1d(second),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(second, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x.transpose(1, 2))
        return self.classifier(self.pool(features).squeeze(-1))
