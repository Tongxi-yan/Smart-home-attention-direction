"""Hardware-neutral real-time inference and CSV replay."""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data_preparation.schema import CLASS_NAMES, COORDINATE_COLUMNS, NODE_NAMES
from source_code.config import FeatureConfig, RealtimeConfig
from source_code.feature_generation.build_features import (
    build_frame_features,
    window_to_graph,
)


@dataclass(frozen=True)
class PredictionResult:
    frame_number: int
    class_index: int
    class_name: str
    confidence: float
    raw_probabilities: list[float]
    smoothed_probabilities: list[float]


class PredictionSmoother:
    """Probability averaging, confidence gating, and short state holding."""

    def __init__(
        self,
        *,
        window: int = 5,
        confidence_threshold: float = 0.60,
        hold_windows: int = 2,
        initial_class: int = 0,
    ) -> None:
        if window < 1 or hold_windows < 0:
            raise ValueError("window must be positive and hold_windows non-negative")
        self.history: deque[np.ndarray] = deque(maxlen=window)
        self.confidence_threshold = confidence_threshold
        self.hold_windows = hold_windows
        self.current_class = initial_class
        self.pending_class: int | None = None
        self.pending_count = 0

    def update(self, probabilities: np.ndarray) -> tuple[int, np.ndarray]:
        values = np.asarray(probabilities, dtype=np.float64)
        if values.shape != (3,) or np.any(values < 0):
            raise ValueError("probabilities must be a non-negative three-value vector")
        total = float(values.sum())
        if total <= 0:
            raise ValueError("probabilities must have a positive sum")
        self.history.append(values / total)
        average = np.mean(np.stack(self.history), axis=0)
        candidate = int(np.argmax(average))
        confidence = float(average[candidate])
        if candidate == self.current_class:
            self.pending_class = None
            self.pending_count = 0
            return self.current_class, average
        if confidence < self.confidence_threshold:
            return self.current_class, average
        if self.pending_class != candidate:
            self.pending_class = candidate
            self.pending_count = 1
        else:
            self.pending_count += 1
        required = max(self.hold_windows, 1)
        if self.pending_count >= required:
            self.current_class = candidate
            self.pending_class = None
            self.pending_count = 0
        return self.current_class, average


def compensate_side_view(
    coordinates: np.ndarray, *, threshold: float | None, gain: float
) -> np.ndarray:
    """Optionally amplify arm offsets when shoulder separation becomes very small.

    The thesis describes this compensation but does not preserve its tuned threshold.
    It is therefore disabled by default and must be calibrated for a new camera setup.
    """

    values = np.asarray(coordinates, dtype=np.float32).copy()
    if threshold is None:
        return values
    index = {name: position for position, name in enumerate(NODE_NAMES)}
    separation = np.linalg.norm(
        values[index["left_shoulder"]] - values[index["right_shoulder"]]
    )
    if separation >= threshold:
        return values
    root = values[index["spine_navel"]]
    for name in (
        "left_elbow",
        "left_wrist",
        "left_hand",
        "left_handtip",
        "right_elbow",
        "right_wrist",
        "right_hand",
        "right_handtip",
    ):
        node = index[name]
        values[node] = root + (1.0 + gain) * (values[node] - root)
    return values


class RealtimeAttentionEngine:
    """Incremental inference engine accepting one canonical 16-node frame at a time."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        feature_config: FeatureConfig | None = None,
        realtime_config: RealtimeConfig | None = None,
        window_size: int = 20,
        stride: int = 3,
        device: str = "cpu",
    ) -> None:
        try:
            import torch
            from .load_model import load_final_model
        except ImportError as error:
            raise RuntimeError("Install the model dependencies: pip install -e '.[models]'") from error

        self.torch = torch
        self.device = torch.device(device)
        self.model, self.checkpoint = load_final_model(checkpoint, device=self.device)
        self.feature_config = feature_config or FeatureConfig()
        self.realtime_config = realtime_config or RealtimeConfig()
        self.window_size = window_size
        self.stride = stride
        self.coordinate_buffer: deque[np.ndarray] = deque(maxlen=window_size + 1)
        self.previous_ema: np.ndarray | None = None
        self.frame_number = 0
        self.smoother = PredictionSmoother(
            window=self.realtime_config.probability_window,
            confidence_threshold=self.realtime_config.confidence_threshold,
            hold_windows=self.realtime_config.hold_windows,
        )

        expected_channels = int(
            self.checkpoint.get("config", {}).get("channels_per_node", 21)
        )
        dummy = np.zeros((window_size, len(NODE_NAMES), 3), dtype=np.float32)
        feature_values, _ = build_frame_features(dummy, self.feature_config)
        actual_channels = int(window_to_graph(feature_values).shape[0])
        if actual_channels != expected_channels:
            raise ValueError(
                "Feature configuration is incompatible with the checkpoint: "
                f"checkpoint expects {expected_channels} channels/node, "
                f"pipeline produces {actual_channels}"
            )

    def add_frame(self, coordinates: np.ndarray) -> PredictionResult | None:
        values = np.asarray(coordinates, dtype=np.float32)
        if values.shape != (len(NODE_NAMES), 3):
            raise ValueError(f"Expected [16, 3], got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("Coordinates must be finite before real-time inference")
        values = compensate_side_view(
            values,
            threshold=self.realtime_config.side_view_threshold,
            gain=self.realtime_config.side_view_gain,
        )
        alpha = self.realtime_config.feature_ema_alpha
        if self.previous_ema is None:
            smoothed = values
        else:
            smoothed = alpha * values + (1.0 - alpha) * self.previous_ema
        self.previous_ema = smoothed
        self.coordinate_buffer.append(smoothed.copy())
        self.frame_number += 1
        if len(self.coordinate_buffer) < self.window_size:
            return None
        if (self.frame_number - self.window_size) % self.stride != 0:
            return None

        coordinates_with_context = np.stack(self.coordinate_buffer)
        features, _ = build_frame_features(
            coordinates_with_context, self.feature_config
        )
        features = features[-self.window_size :]
        graph = window_to_graph(features)[None]
        tensor = self.torch.tensor(graph, dtype=self.torch.float32, device=self.device)
        with self.torch.no_grad():
            logits = self.model(tensor)
            raw = self.torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        class_index, averaged = self.smoother.update(raw)
        return PredictionResult(
            frame_number=self.frame_number,
            class_index=class_index,
            class_name=CLASS_NAMES[class_index],
            confidence=float(averaged[class_index]),
            raw_probabilities=raw.astype(float).tolist(),
            smoothed_probabilities=averaged.astype(float).tolist(),
        )


def replay_csv(
    csv_path: str | Path,
    checkpoint: str | Path,
    *,
    realtime_config: RealtimeConfig | None = None,
    device: str = "cpu",
) -> list[PredictionResult]:
    frame = pd.read_csv(csv_path, encoding="utf-8-sig")
    missing = [column for column in COORDINATE_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"CSV is missing canonical coordinate columns: {missing}")
    engine = RealtimeAttentionEngine(
        checkpoint,
        realtime_config=realtime_config,
        device=device,
    )
    results: list[PredictionResult] = []
    values = frame[list(COORDINATE_COLUMNS)].to_numpy(dtype=np.float32).reshape(
        len(frame), len(NODE_NAMES), 3
    )
    for coordinates in values:
        result = engine.add_frame(coordinates)
        if result is not None:
            results.append(result)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Canonical processed CSV")
    parser.add_argument("--checkpoint", required=True, help="Final .pth checkpoint")
    parser.add_argument("--output", help="Optional JSON Lines output")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, or mps")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = replay_csv(args.input, args.checkpoint, device=args.device)
    lines = [json.dumps(asdict(result), ensure_ascii=False) for result in results]
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as stream:
            stream.write("\n".join(lines))
            if lines:
                stream.write("\n")
    else:
        for line in lines:
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
