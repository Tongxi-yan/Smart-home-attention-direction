"""Checkpoint-compatible GCN + joint attention + BiLSTM classifier.

Attribute names and layer ordering intentionally match the saved project checkpoint.
Changing them breaks ``state_dict`` compatibility.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class AttentionGraph:
    def __init__(self) -> None:
        self.num_nodes = 16
        self.edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (1, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            (1, 9),
            (9, 10),
            (10, 11),
            (11, 12),
            (12, 13),
            (1, 14),
            (2, 14),
            (3, 14),
            (7, 14),
            (8, 14),
            (12, 14),
            (13, 14),
            (1, 15),
            (2, 15),
            (3, 15),
            (7, 15),
            (8, 15),
            (12, 15),
            (13, 15),
        ]
        self.A = self._build_adjacency_matrix()

    def _build_adjacency_matrix(self) -> torch.Tensor:
        adjacency = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        np.fill_diagonal(adjacency, 1.0)
        for source, target in self.edges:
            adjacency[source, target] = 1.0
            adjacency[target, source] = 1.0
        degree = adjacency.sum(axis=1)
        inverse_sqrt = np.power(degree, -0.5).flatten()
        inverse_sqrt[np.isinf(inverse_sqrt)] = 0.0
        normalized = np.diag(inverse_sqrt) @ adjacency @ np.diag(inverse_sqrt)
        return torch.tensor(normalized, dtype=torch.float32)


class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_nodes: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.edge_importance = nn.Parameter(torch.ones(num_nodes, num_nodes))
        self.adaptive_A = nn.Parameter(torch.zeros(num_nodes, num_nodes))
        self.adaptive_scale = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))

    def effective_adjacency(self, adjacency: torch.Tensor) -> torch.Tensor:
        adaptive = torch.tanh(self.adaptive_A) * self.adaptive_scale
        effective = adjacency * self.edge_importance + adaptive
        return torch.softmax(effective, dim=-1)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        projected = self.proj(x)
        effective = self.effective_adjacency(adjacency)
        return torch.einsum("nctv,vw->nctw", projected, effective).contiguous()


class TemporalConvBranch(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("temporal kernel size must be odd")
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=(kernel_size, 1),
                padding=(padding, 0),
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiScaleTemporalConv(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_sizes: tuple[int, ...] = (3, 5),
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        if not kernel_sizes:
            raise ValueError("kernel_sizes cannot be empty")
        self.branches = nn.ModuleList(
            [TemporalConvBranch(channels, size) for size in kernel_sizes]
        )
        self.branch_logits = nn.Parameter(torch.zeros(len(kernel_sizes)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = torch.stack([branch(x) for branch in self.branches], dim=0)
        weights = torch.softmax(self.branch_logits, dim=0).view(-1, 1, 1, 1, 1)
        return self.dropout(F.relu(torch.sum(outputs * weights, dim=0), inplace=True))


class STGCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_nodes: int,
        temporal_kernel_sizes: tuple[int, ...] = (3, 5),
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        self.gcn = SpatialGraphConv(in_channels, out_channels, num_nodes)
        self.spatial_bn = nn.BatchNorm2d(out_channels)
        self.temporal_conv = MultiScaleTemporalConv(
            out_channels, temporal_kernel_sizes, dropout
        )
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        output = self.gcn(x, adjacency)
        output = F.relu(self.spatial_bn(output), inplace=True)
        output = self.temporal_conv(output)
        return F.relu(output + residual, inplace=True)


class JointScoreMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class TemporalScoreMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class GCNJointAttentionLSTMClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 3,
        gcn_hidden_dims: tuple[int, int, int] = (64, 128, 256),
        attn_hidden_dim: int = 128,
        lstm_hidden_dim: int = 128,
        lstm_layers: int = 1,
        bidirectional: bool = True,
        classifier_hidden_dim: int = 128,
        dropout: float = 0.35,
        temporal_pool: str = "mean",
        temporal_kernel_sizes: tuple[int, ...] = (3, 5),
        joint_attention_dropout: float = 0.10,
        joint_attention_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        graph = AttentionGraph()
        self.num_nodes = graph.num_nodes
        self.temporal_pool = temporal_pool
        self.bidirectional = bidirectional
        self.joint_attention_temperature = max(
            float(joint_attention_temperature), 1e-3
        )
        self.register_buffer("A", graph.A)

        dimensions = [in_channels, *gcn_hidden_dims]
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

        backbone_dim = dimensions[-1]
        self.joint_scorer = JointScoreMLP(
            backbone_dim * 2, attn_hidden_dim, min(0.30, dropout)
        )
        self.joint_attention_dropout = nn.Dropout(joint_attention_dropout)
        self.lstm = nn.LSTM(
            input_size=backbone_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        sequence_dim = lstm_hidden_dim * (2 if bidirectional else 1)
        self.frame_skip_proj = nn.Linear(backbone_dim, sequence_dim)
        self.sequence_norm = nn.LayerNorm(sequence_dim)

        if temporal_pool == "attn":
            self.temporal_scorer = TemporalScoreMLP(
                sequence_dim * 2, attn_hidden_dim, min(0.30, dropout)
            )
        elif temporal_pool != "mean":
            raise ValueError("temporal_pool must be 'mean' or 'attn'")

        self.classifier = nn.Sequential(
            nn.LayerNorm(sequence_dim),
            nn.Linear(sequence_dim, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected [N, C, T, V], got {tuple(x.shape)}")
        batch, channels, time, nodes = x.shape
        if nodes != self.num_nodes:
            raise ValueError(
                f"unexpected num_nodes: got {nodes}, expected {self.num_nodes}"
            )
        normalized = x.permute(0, 3, 1, 2).contiguous().view(
            batch, nodes * channels, time
        )
        normalized = self.data_bn(normalized)
        return normalized.view(batch, nodes, channels, time).permute(
            0, 2, 3, 1
        ).contiguous()

    def _joint_attention_pool(
        self, joint_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context = joint_features.mean(dim=2, keepdim=True).expand(
            -1, -1, self.num_nodes, -1
        )
        scores = self.joint_scorer(torch.cat((joint_features, context), dim=-1))
        scores = scores / self.joint_attention_temperature
        attention = torch.softmax(scores, dim=-1)
        attention = self.joint_attention_dropout(attention)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        frames = torch.sum(joint_features * attention.unsqueeze(-1), dim=2)
        return frames, attention, scores

    def _temporal_pool(
        self, sequence: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.temporal_pool == "mean":
            return sequence.mean(dim=1), None
        context = sequence.mean(dim=1, keepdim=True).expand(-1, sequence.size(1), -1)
        logits = self.temporal_scorer(torch.cat((sequence, context), dim=-1))
        attention = torch.softmax(logits, dim=-1)
        return torch.sum(sequence * attention.unsqueeze(-1), dim=1), attention

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        output = self._normalize_input(x)
        for layer in self.gcn_layers:
            output = layer(output, self.A)
        joints = output.permute(0, 2, 3, 1).contiguous()
        frames, joint_attention, joint_scores = self._joint_attention_pool(joints)
        sequence, _ = self.lstm(frames)
        sequence = self.sequence_norm(sequence + self.frame_skip_proj(frames))
        global_feature, temporal_attention = self._temporal_pool(sequence)
        logits = self.classifier(global_feature)
        if not return_attention:
            return logits
        return {
            "logits": logits,
            "joint_attention": joint_attention,
            "joint_score_logits": joint_scores,
            "frame_features": frames,
            "lstm_features": sequence,
            "global_feature": global_feature,
            "temporal_attention": temporal_attention,
            "effective_adjacency": self.gcn_layers[0].gcn.effective_adjacency(self.A),
        }
