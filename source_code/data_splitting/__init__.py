"""Dispatch between the separate window-level and segment-level protocols."""

from __future__ import annotations

import numpy as np

from ..feature_generation.create_windows import WindowDataset
from .common import SplitIndices, fixed_test_pool, load_window_keys, save_window_keys
from .segment import assert_segment_disjoint, split_segment_level
from .segment import split_fixed_test_pool as split_fixed_segment_pool
from .window import split_fixed_test_pool as split_fixed_window_pool
from .window import split_window_level


def split_dataset(
    data: WindowDataset,
    *,
    strategy: str,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> SplitIndices:
    if strategy == "window":
        return split_window_level(data, ratios=ratios, seed=seed)
    if strategy == "segment":
        return split_segment_level(data, ratios=ratios, seed=seed)
    raise ValueError("strategy must be 'window' or 'segment'")


def split_with_fixed_test(
    data: WindowDataset,
    *,
    strategy: str,
    test_keys: set[tuple[int, int, int]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> SplitIndices:
    """Keep an externally fixed test set and split only the remaining pool."""

    if train_ratio < 0 or val_ratio < 0 or train_ratio + val_ratio <= 0:
        raise ValueError("train_ratio and val_ratio must define a non-empty pool")
    test, remaining, pool = fixed_test_pool(data, test_keys)
    train_portion = train_ratio / (train_ratio + val_ratio)
    rng = np.random.default_rng(seed)
    if strategy == "window":
        pool_train, pool_validation = split_fixed_window_pool(
            pool, train_portion=train_portion, rng=rng
        )
    elif strategy == "segment":
        pool_train, pool_validation = split_fixed_segment_pool(
            pool, train_portion=train_portion, rng=rng
        )
    else:
        raise ValueError("strategy must be 'window' or 'segment'")
    return SplitIndices(
        train=remaining[pool_train],
        validation=remaining[pool_validation],
        test=test,
    )


__all__ = [
    "SplitIndices",
    "assert_segment_disjoint",
    "load_window_keys",
    "save_window_keys",
    "split_dataset",
    "split_segment_level",
    "split_window_level",
    "split_with_fixed_test",
]
