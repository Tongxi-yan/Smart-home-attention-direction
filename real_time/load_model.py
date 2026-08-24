"""Load the final model from a self-describing PyTorch checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from source_code.model_experiments.gcn_attention_bilstm import (
    GCNJointAttentionLSTMClassifier,
)


def _torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch < 2.0
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a dictionary")
    return checkpoint


def load_final_model(
    checkpoint_path: str | Path, *, device: str | torch.device = "cpu"
) -> tuple[GCNJointAttentionLSTMClassifier, dict[str, Any]]:
    target = torch.device(device)
    checkpoint = _torch_load(Path(checkpoint_path), target)
    state = checkpoint.get("model_state_dict", checkpoint)
    config = checkpoint.get("config", {})
    model_config = dict(config.get("model", {}))
    channels = int(config.get("channels_per_node", 21))
    if int(config.get("num_nodes", 16)) != 16:
        raise ValueError("This package supports the final 16-node graph only")
    allowed = {
        "gcn_hidden_dims",
        "attn_hidden_dim",
        "temporal_kernel_sizes",
        "lstm_hidden_dim",
        "lstm_layers",
        "bidirectional",
        "classifier_hidden_dim",
        "dropout",
        "joint_attention_dropout",
        "joint_attention_temperature",
        "temporal_pool",
    }
    model_config = {key: value for key, value in model_config.items() if key in allowed}
    for key in ("gcn_hidden_dims", "temporal_kernel_sizes"):
        if key in model_config:
            model_config[key] = tuple(model_config[key])
    model = GCNJointAttentionLSTMClassifier(
        in_channels=channels, num_classes=3, **model_config
    ).to(target)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, checkpoint
