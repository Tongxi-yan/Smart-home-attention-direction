"""Segment-level dataset partitioning."""

from __future__ import annotations

import numpy as np

from ...feature_generation.create_windows import WindowDataset
from ..common import SplitIndices, allocate, validate_ratios


def split_segment_level(
    data: WindowDataset,
    *,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
) -> SplitIndices:
    """Stratified group split where all windows from one segment stay together."""

    validate_ratios(ratios)
    rng = np.random.default_rng(seed)
    groups: dict[tuple[int, int], list[int]] = {}
    for index, key in enumerate(data.segment_keys()):
        groups.setdefault(key, []).append(index)

    by_label: dict[int, list[tuple[int, int]]] = {}
    for key, indices in groups.items():
        labels, counts = np.unique(data.y[indices], return_counts=True)
        label = int(labels[np.argmax(counts)])
        by_label.setdefault(label, []).append(key)

    selected: list[list[int]] = [[], [], []]
    for label in sorted(by_label):
        group_keys = by_label[label]
        rng.shuffle(group_keys)
        counts = allocate(len(group_keys), ratios)
        boundaries = (counts[0], counts[0] + counts[1])
        partitions = (
            group_keys[: boundaries[0]],
            group_keys[boundaries[0] : boundaries[1]],
            group_keys[boundaries[1] :],
        )
        for target, partition in zip(selected, partitions):
            for key in partition:
                target.extend(groups[key])

    arrays = [np.asarray(indices, dtype=np.int64) for indices in selected]
    for indices in arrays:
        rng.shuffle(indices)
    return SplitIndices(arrays[0], arrays[1], arrays[2])


def split_fixed_test_pool(
    pool: WindowDataset, *, train_portion: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    groups: dict[tuple[int, int], list[int]] = {}
    for index, key in enumerate(pool.segment_keys()):
        groups.setdefault(key, []).append(index)
    by_label: dict[int, list[tuple[int, int]]] = {}
    for key, indices in groups.items():
        labels, counts = np.unique(pool.y[indices], return_counts=True)
        label = int(labels[np.argmax(counts)])
        by_label.setdefault(label, []).append(key)
    train_groups: set[tuple[int, int]] = set()
    validation_groups: set[tuple[int, int]] = set()
    for group_keys in by_label.values():
        rng.shuffle(group_keys)
        train_count = int(len(group_keys) * train_portion)
        train_groups.update(group_keys[:train_count])
        validation_groups.update(group_keys[train_count:])
    pool_keys = pool.segment_keys()
    pool_train = np.asarray(
        [index for index, key in enumerate(pool_keys) if key in train_groups],
        dtype=np.int64,
    )
    pool_validation = np.asarray(
        [index for index, key in enumerate(pool_keys) if key in validation_groups],
        dtype=np.int64,
    )
    return pool_train, pool_validation


def assert_segment_disjoint(data: WindowDataset, split: SplitIndices) -> None:
    partitions = []
    keys = data.segment_keys()
    for indices in (split.train, split.validation, split.test):
        partitions.append({keys[int(index)] for index in indices})
    if (
        partitions[0] & partitions[1]
        or partitions[0] & partitions[2]
        or partitions[1] & partitions[2]
    ):
        raise AssertionError("A segment appears in more than one split")
