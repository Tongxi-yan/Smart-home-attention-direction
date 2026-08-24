"""State construction and sliding-window generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import FeatureConfig, WindowConfig
from .build_features import frame_features_from_dataframe


@dataclass(frozen=True)
class WindowDataset:
    X: np.ndarray
    y: np.ndarray
    dataset_id: np.ndarray
    window_start: np.ndarray
    window_end: np.ndarray
    segment_id: np.ndarray
    feature_names: tuple[str, ...]

    def __len__(self) -> int:
        return int(len(self.y))

    def subset(self, indices: np.ndarray) -> "WindowDataset":
        idx = np.asarray(indices, dtype=np.int64)
        return WindowDataset(
            X=self.X[idx],
            y=self.y[idx],
            dataset_id=self.dataset_id[idx],
            window_start=self.window_start[idx],
            window_end=self.window_end[idx],
            segment_id=self.segment_id[idx],
            feature_names=self.feature_names,
        )

    @classmethod
    def concatenate(cls, datasets: list["WindowDataset"]) -> "WindowDataset":
        if not datasets:
            raise ValueError("No datasets to concatenate")
        reference = datasets[0].feature_names
        if any(dataset.feature_names != reference for dataset in datasets[1:]):
            raise ValueError("All recordings must use the same feature columns")
        return cls(
            X=np.concatenate([dataset.X for dataset in datasets]),
            y=np.concatenate([dataset.y for dataset in datasets]),
            dataset_id=np.concatenate([dataset.dataset_id for dataset in datasets]),
            window_start=np.concatenate([dataset.window_start for dataset in datasets]),
            window_end=np.concatenate([dataset.window_end for dataset in datasets]),
            segment_id=np.concatenate([dataset.segment_id for dataset in datasets]),
            feature_names=reference,
        )

    def window_keys(self) -> list[tuple[int, int, int]]:
        return [
            (int(dataset), int(start), int(end))
            for dataset, start, end in zip(
                self.dataset_id, self.window_start, self.window_end
            )
        ]

    def segment_keys(self) -> list[tuple[int, int]]:
        return [
            (int(dataset), int(segment))
            for dataset, segment in zip(self.dataset_id, self.segment_id)
        ]


def state_codes(frame: pd.DataFrame) -> np.ndarray:
    required = ("label_device1", "label_device2")
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Missing label columns: {missing}")
    device1 = pd.to_numeric(frame["label_device1"], errors="raise").to_numpy(dtype=int)
    device2 = pd.to_numeric(frame["label_device2"], errors="raise").to_numpy(dtype=int)
    if not set(np.unique(device1)).issubset({0, 1}):
        raise ValueError("label_device1 must be binary")
    if not set(np.unique(device2)).issubset({0, 1}):
        raise ValueError("label_device2 must be binary")
    return (device1 + 2 * device2).astype(np.int64)


def build_segment_ids(states: np.ndarray) -> np.ndarray:
    values = np.asarray(states, dtype=np.int64)
    if values.ndim != 1:
        raise ValueError("states must be one-dimensional")
    result = np.zeros(len(values), dtype=np.int64)
    if len(values) > 1:
        result[1:] = np.cumsum(values[1:] != values[:-1])
    return result


def _majority_label(states: np.ndarray) -> tuple[int, float]:
    valid = states[np.isin(states, (0, 1, 2))]
    if len(valid) == 0:
        return 3, 0.0
    counts = np.bincount(valid, minlength=3)
    label = int(np.argmax(counts))
    return label, float(counts[label] / len(states))


def build_windows(
    frame: pd.DataFrame,
    *,
    dataset_id: int,
    window_config: WindowConfig | None = None,
    feature_config: FeatureConfig | None = None,
) -> WindowDataset:
    window_config = window_config or WindowConfig()
    feature_config = feature_config or FeatureConfig()
    states = state_codes(frame)
    segments = build_segment_ids(states)
    label3 = (
        pd.to_numeric(frame["label_3"], errors="coerce").fillna(0).to_numpy(dtype=int)
        if "label_3" in frame
        else np.zeros(len(frame), dtype=int)
    )
    features, names = frame_features_from_dataframe(
        frame, feature_config, segment_ids=segments
    )
    size = window_config.window_size
    if len(frame) < size:
        return WindowDataset(
            X=np.empty((0, size, features.shape[1]), dtype=np.float32),
            y=np.empty(0, dtype=np.int64),
            dataset_id=np.empty(0, dtype=np.int64),
            window_start=np.empty(0, dtype=np.int64),
            window_end=np.empty(0, dtype=np.int64),
            segment_id=np.empty(0, dtype=np.int64),
            feature_names=tuple(names),
        )

    if window_config.boundary_policy == "segment_only":
        candidate_starts: list[int] = []
        boundaries = np.flatnonzero(np.r_[True, states[1:] != states[:-1], True])
        for boundary_index in range(len(boundaries) - 1):
            segment_start = int(boundaries[boundary_index])
            segment_end = int(boundaries[boundary_index + 1])
            base_starts = np.arange(
                segment_start,
                segment_end - size + 1,
                window_config.stride,
                dtype=np.int64,
            )
            shifted_starts = base_starts - window_config.history_frames
            valid = shifted_starts[
                (shifted_starts >= segment_start)
                & (shifted_starts + size <= segment_end)
            ]
            candidate_starts.extend(valid.tolist())
    else:
        candidate_starts = list(
            range(0, len(frame) - size + 1, window_config.stride)
        )

    windows: list[np.ndarray] = []
    labels: list[int] = []
    starts: list[int] = []
    segment_values: list[int] = []
    for start in candidate_starts:
        end = start + size
        state_window = states[start:end]
        segment_window = segments[start:end]
        if window_config.drop_state_3_windows and np.any(state_window == 3):
            continue

        if window_config.boundary_policy == "segment_only":
            if segment_window[0] != segment_window[-1]:
                continue
            label = int(state_window[0])
            segment = int(segment_window[0])
        elif window_config.boundary_policy == "majority":
            label, purity = _majority_label(state_window)
            if purity < window_config.majority_purity_threshold:
                continue
            center = start + size // 2
            segment = int(segments[min(center, len(segments) - 1)])
        else:
            raise ValueError(
                "boundary_policy must be 'segment_only' or 'majority'"
            )

        if label not in (0, 1, 2):
            continue
        if (
            window_config.forbid_label3_in_off_window
            and label == 0
            and np.any(label3[start:end] == 1)
        ):
            continue
        windows.append(features[start:end])
        labels.append(label)
        starts.append(start)
        segment_values.append(segment)

    if windows:
        X = np.stack(windows).astype(np.float32)
    else:
        X = np.empty((0, size, features.shape[1]), dtype=np.float32)
    count = len(labels)
    starts_array = np.asarray(starts, dtype=np.int64)
    return WindowDataset(
        X=X,
        y=np.asarray(labels, dtype=np.int64),
        dataset_id=np.full(count, dataset_id, dtype=np.int64),
        window_start=starts_array,
        window_end=starts_array + size,
        segment_id=np.asarray(segment_values, dtype=np.int64),
        feature_names=tuple(names),
    )


def load_window_directory(
    directory: str | Path,
    *,
    pattern: str = "dataset_*.csv",
    window_config: WindowConfig | None = None,
    feature_config: FeatureConfig | None = None,
) -> WindowDataset:
    source = Path(directory)
    paths = sorted(source.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching {pattern!r} in {source}")
    datasets: list[WindowDataset] = []
    for sequence, path in enumerate(paths, start=1):
        digits = "".join(character for character in path.stem if character.isdigit())
        dataset_id = int(digits) if digits else sequence
        frame = pd.read_csv(path, encoding="utf-8-sig")
        dataset = build_windows(
            frame,
            dataset_id=dataset_id,
            window_config=window_config,
            feature_config=feature_config,
        )
        if len(dataset):
            datasets.append(dataset)
    if not datasets:
        raise ValueError("All recordings produced zero valid windows")
    return WindowDataset.concatenate(datasets)
