"""ST-GCN baseline from the final experiment set."""

from __future__ import annotations

import torch
from torch import nn

from ..gcn_attention_bilstm.model import AttentionGraph, STGCNBlock


class STGCNClassifier(nn.Module):
    """Spatial-temporal graph baseline without joint attention or BiLSTM."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 3,
        hidden_dims: tuple[int, int, int] = (64, 128, 256),
        temporal_kernel_sizes: tuple[int, ...] = (3, 5),
        dropout: float = 0.35,
        classifier_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        graph = AttentionGraph()
        self.num_nodes = graph.num_nodes
        self.register_buffer("A", graph.A)
        dimensions = [in_channels, *hidden_dims]
        self.data_bn = nn.BatchNorm1d(in_channels * self.num_nodes)
        self.gcn_layers = nn.ModuleList(
            [
                STGCNBlock(
                    dimensions[index],
                    dimensions[index + 1],
                    self.num_nodes,
                    temporal_kernel_sizes,
                    dropout,
                )
                for index in range(len(dimensions) - 1)
            ]
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(dimensions[-1]),
            nn.Linear(dimensions[-1], classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, time, nodes = x.shape
        if nodes != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, received {nodes}")
        output = x.permute(0, 3, 1, 2).contiguous().view(
            batch, nodes * channels, time
        )
        output = self.data_bn(output)
        output = output.view(batch, nodes, channels, time).permute(0, 2, 3, 1)
        for layer in self.gcn_layers:
            output = layer(output, self.A)
        return self.classifier(output.mean(dim=(2, 3)))
