"""Dependency-light metrics for the three-class task."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    confusion_matrix: list[list[int]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, *, num_classes: int = 3
) -> np.ndarray:
    truth = np.asarray(y_true, dtype=int)
    prediction = np.asarray(y_pred, dtype=int)
    if truth.shape != prediction.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for actual, predicted in zip(truth, prediction):
        if not 0 <= actual < num_classes or not 0 <= predicted < num_classes:
            raise ValueError("Labels are outside the configured class range")
        matrix[actual, predicted] += 1
    return matrix


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, *, num_classes: int = 3
) -> ClassificationMetrics:
    matrix = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    total = int(matrix.sum())
    accuracy = float(np.trace(matrix) / total) if total else 0.0
    recalls: list[float] = []
    f1_scores: list[float] = []
    for label in range(num_classes):
        true_positive = int(matrix[label, label])
        false_negative = int(matrix[label].sum() - true_positive)
        false_positive = int(matrix[:, label].sum() - true_positive)
        recall_denominator = true_positive + false_negative
        recalls.append(
            true_positive / recall_denominator if recall_denominator else 0.0
        )
        f1_denominator = 2 * true_positive + false_positive + false_negative
        f1_scores.append(
            2 * true_positive / f1_denominator if f1_denominator else 0.0
        )
    return ClassificationMetrics(
        accuracy=accuracy,
        balanced_accuracy=float(np.mean(recalls)),
        macro_f1=float(np.mean(f1_scores)),
        confusion_matrix=matrix.tolist(),
    )
