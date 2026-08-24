"""Root-relative graph inputs and attention-related geometric features."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..config import FeatureConfig
from data_preparation.schema import COORDINATE_COLUMNS, NODE_NAMES


def _safe_norm(values: np.ndarray, axis: int = -1, epsilon: float = 1e-6) -> np.ndarray:
    return np.sqrt(np.sum(values * values, axis=axis) + epsilon)


def _first_difference(
    values: np.ndarray, segment_ids: np.ndarray | None = None
) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float32)
    if len(values) > 1:
        result[1:] = values[1:] - values[:-1]
    if segment_ids is not None and len(values) > 1:
        segment_ids = np.asarray(segment_ids, dtype=np.int64)
        if segment_ids.shape != (len(values),):
            raise ValueError("segment_ids must contain one value per frame")
        result[1:][segment_ids[1:] != segment_ids[:-1]] = 0.0
    return result


def _validate_coordinates(coordinates: np.ndarray) -> np.ndarray:
    values = np.asarray(coordinates, dtype=np.float32)
    if values.ndim == 3 and values.shape[1:] == (len(NODE_NAMES), 3):
        return values
    if values.ndim == 2 and values.shape[1] == len(COORDINATE_COLUMNS):
        return values.reshape(len(values), len(NODE_NAMES), 3)
    raise ValueError(
        "coordinates must have shape [frames, 48] or [frames, 16, 3]; "
        f"received {values.shape}"
    )


def feature_names(config: FeatureConfig) -> list[str]:
    names: list[str] = []
    for joint in config.distance_joints:
        if config.use_distance_to_device1:
            names.append(f"dist_{joint}_to_device1")
        if config.use_distance_to_device2:
            names.append(f"dist_{joint}_to_device2")
        if config.use_distance_diff:
            names.append(f"distdiff_{joint}_device1_minus_device2")
    if config.use_horizontal_head_direction:
        names.extend(
            (
                "head_dir_horizontal_cos_to_device1",
                "head_dir_horizontal_cos_to_device2",
            )
        )
    if config.use_velocity:
        names.extend(f"velocity_{name}" for name in list(names))
    return names


def build_engineered_features(
    coordinates: np.ndarray,
    config: FeatureConfig | None = None,
    *,
    segment_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Return global geometric features for every frame."""

    config = config or FeatureConfig()
    coords = _validate_coordinates(coordinates)
    index = {name: idx for idx, name in enumerate(NODE_NAMES)}
    device1 = coords[:, index["device1"]]
    device2 = coords[:, index["device2"]]

    parts: list[np.ndarray] = []
    names: list[str] = []
    for joint in config.distance_joints:
        if joint not in index:
            raise ValueError(f"Unknown distance joint: {joint}")
        point = coords[:, index[joint]]
        distance1 = _safe_norm(point - device1, axis=-1)
        distance2 = _safe_norm(point - device2, axis=-1)
        if config.use_distance_to_device1:
            parts.append(distance1[:, None])
            names.append(f"dist_{joint}_to_device1")
        if config.use_distance_to_device2:
            parts.append(distance2[:, None])
            names.append(f"dist_{joint}_to_device2")
        if config.use_distance_diff:
            parts.append((distance1 - distance2)[:, None])
            names.append(f"distdiff_{joint}_device1_minus_device2")

    if config.use_horizontal_head_direction:
        if config.vertical_axis not in (0, 1, 2):
            raise ValueError("vertical_axis must be 0, 1, or 2")
        nose = coords[:, index["nose"]]
        neck = coords[:, index["neck"]]
        head_direction = nose - neck
        to_devices = (device1 - nose, device2 - nose)
        head_direction = head_direction.copy()
        head_direction[:, config.vertical_axis] = 0.0
        for device_number, target in enumerate(to_devices, start=1):
            target = target.copy()
            target[:, config.vertical_axis] = 0.0
            cosine = np.sum(head_direction * target, axis=-1) / (
                _safe_norm(head_direction) * _safe_norm(target) + 1e-6
            )
            parts.append(cosine[:, None])
            names.append(f"head_dir_horizontal_cos_to_device{device_number}")

    if not parts:
        return np.empty((len(coords), 0), dtype=np.float32), []

    static = np.concatenate(parts, axis=1).astype(np.float32)
    if not config.use_velocity:
        return static, names
    velocity = _first_difference(static, segment_ids=segment_ids)
    velocity_names = [f"velocity_{name}" for name in names]
    return np.concatenate((static, velocity), axis=1), names + velocity_names


def build_frame_features(
    coordinates: np.ndarray,
    config: FeatureConfig | None = None,
    *,
    segment_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    coords = _validate_coordinates(coordinates)
    flat = coords.reshape(len(coords), -1).astype(np.float32)
    config = config or FeatureConfig()
    if not config.enabled:
        return flat, list(COORDINATE_COLUMNS)
    engineered, engineered_names = build_engineered_features(
        coords, config, segment_ids=segment_ids
    )
    return np.concatenate((flat, engineered), axis=1), [
        *COORDINATE_COLUMNS,
        *engineered_names,
    ]


def frame_features_from_dataframe(
    frame: pd.DataFrame,
    config: FeatureConfig | None = None,
    *,
    segment_ids: Sequence[int] | None = None,
) -> tuple[np.ndarray, list[str]]:
    missing = [column for column in COORDINATE_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Missing coordinate columns: {missing}")
    ids = None if segment_ids is None else np.asarray(segment_ids, dtype=np.int64)
    return build_frame_features(
        frame[list(COORDINATE_COLUMNS)].to_numpy(dtype=np.float32),
        config,
        segment_ids=ids,
    )


def window_to_graph(window: np.ndarray, *, num_nodes: int = 16) -> np.ndarray:
    """Convert one ``[T, 48+E]`` window to ``[3+E, T, 16]``."""

    values = np.asarray(window, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] < num_nodes * 3:
        raise ValueError(f"Expected [T, >= {num_nodes * 3}], got {values.shape}")
    coordinate_width = num_nodes * 3
    coordinates = values[:, :coordinate_width].reshape(len(values), num_nodes, 3).copy()
    coordinates -= coordinates[:, 0:1, :]
    extras = values[:, coordinate_width:]
    if extras.shape[1]:
        broadcast = np.repeat(extras[:, None, :], num_nodes, axis=1)
        node_features = np.concatenate((coordinates, broadcast), axis=2)
    else:
        node_features = coordinates
    return np.transpose(node_features, (2, 0, 1)).astype(np.float32)


def windows_to_graph(windows: np.ndarray) -> np.ndarray:
    values = np.asarray(windows, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"Expected [N, T, F], got {values.shape}")
    return np.stack([window_to_graph(window) for window in values], axis=0)
