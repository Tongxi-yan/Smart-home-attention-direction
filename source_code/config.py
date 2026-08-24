"""Typed project configuration and JSON loading helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar


@dataclass
class WindowConfig:
    window_size: int = 20
    stride: int = 3
    history_frames: int = 5
    boundary_policy: Literal["segment_only", "majority"] = "segment_only"
    majority_purity_threshold: float = 0.5
    drop_state_3_windows: bool = True
    forbid_label3_in_off_window: bool = True


@dataclass
class FeatureConfig:
    """Feature route used by the discoverable final checkpoint.

    The shipped checkpoint uses three root-relative coordinate channels plus 18
    global engineered channels (nine distances/differences and their first-order
    differences), broadcast over every graph node: 21 channels per node in total.
    """

    enabled: bool = True
    distance_joints: tuple[str, ...] = ("head", "left_hand", "right_hand")
    use_distance_to_device1: bool = True
    use_distance_to_device2: bool = True
    use_distance_diff: bool = True
    use_horizontal_head_direction: bool = False
    use_velocity: bool = True
    vertical_axis: int = 1


@dataclass
class SplitConfig:
    strategy: Literal["window", "segment"] = "window"
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    fixed_test_windows: str | None = None


@dataclass
class ModelConfig:
    architecture: Literal[
        "xgboost", "cnn", "bilstm", "stgcn", "gcn_attention_bilstm"
    ] = (
        "gcn_attention_bilstm"
    )
    gcn_hidden_dims: tuple[int, int, int] = (64, 128, 256)
    attention_hidden_dim: int = 128
    temporal_kernel_sizes: tuple[int, ...] = (3, 5)
    lstm_hidden_dim: int = 128
    lstm_layers: int = 1
    bidirectional: bool = True
    classifier_hidden_dim: int = 128
    dropout: float = 0.35
    joint_attention_dropout: float = 0.10
    joint_attention_temperature: float = 1.0
    temporal_pool: Literal["mean", "attn"] = "mean"


@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 80
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 10
    gradient_clip: float = 1.0
    class_weighted_loss: bool = True
    device: str = "auto"


@dataclass
class RealtimeConfig:
    feature_ema_alpha: float = 0.20
    probability_window: int = 5
    confidence_threshold: float = 0.60
    hold_windows: int = 2
    side_view_threshold: float | None = None
    side_view_gain: float = 0.25


@dataclass
class ProjectConfig:
    windows: WindowConfig = field(default_factory=WindowConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    realtime: RealtimeConfig = field(default_factory=RealtimeConfig)

    def validate(self) -> None:
        if self.windows.window_size < 1:
            raise ValueError("window_size must be positive")
        if self.windows.stride < 1:
            raise ValueError("stride must be positive")
        if not 0.0 < self.windows.majority_purity_threshold <= 1.0:
            raise ValueError("majority_purity_threshold must be in (0, 1]")
        ratios = (
            self.split.train_ratio,
            self.split.val_ratio,
            self.split.test_ratio,
        )
        if any(value < 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-8:
            raise ValueError("train/val/test ratios must be non-negative and sum to 1")
        if not 0.0 < self.realtime.feature_ema_alpha <= 1.0:
            raise ValueError("feature_ema_alpha must be in (0, 1]")
        if self.realtime.probability_window < 1:
            raise ValueError("probability_window must be positive")
        if not 0.0 <= self.realtime.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")
        if self.realtime.hold_windows < 0:
            raise ValueError("hold_windows must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


T = TypeVar("T")


def _update_dataclass(instance: T, values: dict[str, Any]) -> T:
    known = {item.name: item for item in fields(instance)}
    unknown = sorted(set(values) - set(known))
    if unknown:
        raise ValueError(
            f"Unknown configuration keys for {type(instance).__name__}: {unknown}"
        )
    for name, value in values.items():
        current = getattr(instance, name)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"Configuration section {name!r} must be an object")
            _update_dataclass(current, value)
        elif isinstance(current, tuple) and isinstance(value, list):
            setattr(instance, name, tuple(value))
        else:
            setattr(instance, name, value)
    return instance


def load_config(path: str | Path | None = None) -> ProjectConfig:
    config = ProjectConfig()
    if path is not None:
        config_path = Path(path).resolve()
        with config_path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict):
            raise ValueError("The configuration root must be a JSON object")
        _update_dataclass(config, data)
        if config.split.fixed_test_windows:
            fixed_path = Path(config.split.fixed_test_windows)
            if not fixed_path.is_absolute():
                config.split.fixed_test_windows = str(
                    (config_path.parent / fixed_path).resolve()
                )
    config.validate()
    return config


def save_config(config: ProjectConfig, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(config.to_dict(), stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return output
