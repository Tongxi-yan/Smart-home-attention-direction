"""Window-level dataset partitioning."""

from __future__ import annotations

import numpy as np

from ...feature_generation.create_windows import WindowDataset
from ..common import SplitIndices, allocate, validate_ratios


def split_window_level(
    data: WindowDataset,
    *,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> SplitIndices:
    """Stratified random split where individual windows are the split unit."""

    validate_ratios(ratios)
    rng = np.random.default_rng(seed)
    buckets: list[list[np.ndarray]] = [[], [], []]
    for label in sorted(np.unique(data.y)):
        indices = np.flatnonzero(data.y == label)
        rng.shuffle(indices)
        train_count, val_count, test_count = allocate(len(indices), ratios)
        boundaries = (train_count, train_count + val_count)
        buckets[0].append(indices[: boundaries[0]])
        buckets[1].append(indices[boundaries[0] : boundaries[1]])
        buckets[2].append(indices[boundaries[1] : boundaries[1] + test_count])
    result = [
        np.concatenate(parts) if parts else np.empty(0, dtype=int)
        for parts in buckets
    ]
    for indices in result:
        rng.shuffle(indices)
    return SplitIndices(result[0], result[1], result[2])


def split_fixed_test_pool(
    pool: WindowDataset, *, train_portion: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    permutation = rng.permutation(len(pool))
    train_count = int(len(pool) * train_portion)
    return permutation[:train_count], permutation[train_count:]
