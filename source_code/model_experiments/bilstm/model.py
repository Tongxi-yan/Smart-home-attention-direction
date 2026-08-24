"""BiLSTM baseline from the final experiment set."""

from __future__ import annotations

import torch
from torch import nn


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        bidirectional: bool = True,
        dropout: float = 0.3,
        classifier_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        output_dim = hidden_size * (2 if bidirectional else 1)
        self.attention = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.Tanh(),
            nn.Linear(output_dim // 2, 1),
        )
        context_dim = output_dim * 3
        self.classifier = nn.Sequential(
            nn.Linear(context_dim, classifier_hidden_dim),
            nn.LayerNorm(classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, max(classifier_hidden_dim // 2, 1)),
            nn.LayerNorm(max(classifier_hidden_dim // 2, 1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(classifier_hidden_dim // 2, 1), num_classes),
        )

    def forward(
        self, x: torch.Tensor, *, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        sequence, _ = self.lstm(x)
        scores = self.attention(sequence).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        attended = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        context = torch.cat(
            (attended, sequence.mean(dim=1), sequence.max(dim=1).values), dim=-1
        )
        logits = self.classifier(context)
        if return_attention:
            return logits, weights
        return logits
