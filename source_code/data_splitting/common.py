"""Shared split data structures and fixed-test key I/O."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..feature_generation.create_windows import WindowDataset


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def allocate(count: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    raw = np.asarray(ratios, dtype=float) * count
    base = np.floor(raw).astype(int)
    for index in np.argsort(-(raw - base))[: count - int(base.sum())]:
        base[index] += 1
    return int(base[0]), int(base[1]), int(base[2])


def validate_ratios(ratios: tuple[float, float, float]) -> None:
    if any(value < 0 for value in ratios) or not np.isclose(sum(ratios), 1.0):
        raise ValueError("split ratios must be non-negative and sum to 1")


def fixed_test_pool(
    data: WindowDataset, test_keys: set[tuple[int, int, int]]
) -> tuple[np.ndarray, np.ndarray, WindowDataset]:
    keys = data.window_keys()
    test = np.asarray(
        [index for index, key in enumerate(keys) if key in test_keys], dtype=np.int64
    )
    if len(test) != len(test_keys):
        found = {keys[int(index)] for index in test}
        missing = sorted(test_keys - found)
        raise ValueError(f"Fixed test keys not present in current data: {missing[:10]}")
    test_set = set(test.tolist())
    remaining = np.asarray(
        [index for index in range(len(data)) if index not in test_set], dtype=np.int64
    )
    return test, remaining, data.subset(remaining)


def save_window_keys(data: WindowDataset, indices: np.ndarray, path: str | Path) -> Path:
    selected = set(np.asarray(indices, dtype=int).tolist())
    keys = [key for index, key in enumerate(data.window_keys()) if index in selected]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(keys, stream, indent=2)
        stream.write("\n")
    return output


def load_window_keys(path: str | Path) -> set[tuple[int, int, int]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    if isinstance(raw, dict):
        if "window_keys" not in raw:
            raise ValueError("Fixed-test JSON object must contain 'window_keys'")
        raw = raw["window_keys"]
    return {tuple(int(value) for value in key) for key in raw}
