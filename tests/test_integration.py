from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from data_preparation.merge_data import pivot_skeleton_csv
from data_preparation.schema import BODY_JOINT_NAMES


class IntegrationTests(unittest.TestCase):
    def test_long_skeleton_pivot(self):
        rows = []
        for frame in range(3):
            for index, name in enumerate(BODY_JOINT_NAMES):
                rows.append(
                    {
                        "Frame Number": frame,
                        "Body ID": 7,
                        "Joint Name": name.replace("_", " "),
                        "X": index + frame * 0.1,
                        "Y": index + 0.2,
                        "Z": index + 1.0,
                    }
                )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "skeleton.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            result = pivot_skeleton_csv(path)
        self.assertEqual(len(result), 3)
        self.assertIn("spine_navel_x", result)
        self.assertIn("right_handtip_z", result)


if __name__ == "__main__":
    unittest.main()
