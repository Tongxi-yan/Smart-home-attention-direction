from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from data_preparation.schema import (
    COORDINATE_COLUMNS,
    normalize_joint_name,
    state_from_binary_labels,
)
from data_preparation.smooth_data import gaussian_kernel
from real_time import PredictionSmoother
from source_code.config import FeatureConfig, WindowConfig
from source_code.data_splitting import assert_segment_disjoint, split_segment_level
from source_code.feature_generation.build_features import (
    build_frame_features,
    window_to_graph,
)
from source_code.feature_generation.create_windows import build_windows
from source_code.model_experiments.evaluation import calculate_metrics


def synthetic_recording(segment_length: int = 30, repeats: int = 4) -> pd.DataFrame:
    labels = []
    for _ in range(repeats):
        labels.extend([0] * segment_length)
        labels.extend([1] * segment_length)
        labels.extend([2] * segment_length)
    count = len(labels)
    coordinates = np.zeros((count, 16, 3), dtype=np.float32)
    time = np.arange(count, dtype=np.float32)
    for node in range(16):
        coordinates[:, node, 0] = node * 0.05 + time * 0.0001
        coordinates[:, node, 1] = node * -0.02
        coordinates[:, node, 2] = 1.0 + node * 0.01
    frame = pd.DataFrame(coordinates.reshape(count, 48), columns=COORDINATE_COLUMNS)
    frame.insert(0, "frame", np.arange(count))
    frame["label_device1"] = (np.asarray(labels) == 1).astype(int)
    frame["label_device2"] = (np.asarray(labels) == 2).astype(int)
    frame["label_3"] = 0
    return frame


class CorePipelineTests(unittest.TestCase):
    def test_schema_helpers(self):
        self.assertEqual(normalize_joint_name("Spine - Navel"), "spine_navel")
        self.assertEqual(state_from_binary_labels(0, 0), 0)
        self.assertEqual(state_from_binary_labels(1, 0), 1)
        self.assertEqual(state_from_binary_labels(0, 1), 2)

    def test_checkpoint_compatible_feature_shape(self):
        coordinates = np.zeros((20, 16, 3), dtype=np.float32)
        features, names = build_frame_features(coordinates, FeatureConfig())
        self.assertEqual(features.shape, (20, 66))
        self.assertEqual(len(names), 66)
        self.assertEqual(window_to_graph(features).shape, (21, 20, 16))

    def test_gaussian_7_matches_final_kernel_parameters(self):
        kernel = gaussian_kernel(7)
        self.assertEqual(len(kernel), 25)
        self.assertAlmostEqual(float(kernel.sum()), 1.0)
        self.assertTrue(np.allclose(kernel, kernel[::-1]))

    def test_window_and_segment_split(self):
        data = build_windows(
            synthetic_recording(),
            dataset_id=1,
            window_config=WindowConfig(
                window_size=20, stride=5, boundary_policy="segment_only"
            ),
        )
        split = split_segment_level(data, ratios=(0.5, 0.25, 0.25), seed=7)
        self.assertTrue(len(split.train))
        self.assertTrue(len(split.validation))
        self.assertTrue(len(split.test))
        assert_segment_disjoint(data, split)

    def test_majority_policy_keeps_boundary_windows(self):
        frame = synthetic_recording(segment_length=12, repeats=1)
        strict = build_windows(
            frame,
            dataset_id=1,
            window_config=WindowConfig(
                window_size=10, stride=2, boundary_policy="segment_only"
            ),
        )
        majority = build_windows(
            frame,
            dataset_id=1,
            window_config=WindowConfig(
                window_size=10,
                stride=2,
                boundary_policy="majority",
                majority_purity_threshold=0.5,
            ),
        )
        self.assertGreater(len(majority), len(strict))

    def test_prediction_holding(self):
        smoother = PredictionSmoother(window=1, hold_windows=2)
        self.assertEqual(smoother.update(np.array([0.1, 0.8, 0.1]))[0], 0)
        self.assertEqual(smoother.update(np.array([0.1, 0.8, 0.1]))[0], 1)

    def test_metrics(self):
        metrics = calculate_metrics(
            np.array([0, 0, 1, 1, 2, 2]), np.array([0, 0, 1, 2, 2, 2])
        )
        self.assertAlmostEqual(metrics.accuracy, 5 / 6)
        self.assertEqual(metrics.confusion_matrix[1][2], 1)


if __name__ == "__main__":
    unittest.main()
