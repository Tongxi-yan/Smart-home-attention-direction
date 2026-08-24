"""Interpolation, joint selection, Gaussian smoothing, and schema validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import (
    BODY_COORDINATE_COLUMNS,
    COORDINATE_COLUMNS,
    DEVICE_COORDINATE_COLUMNS,
    LABEL_COLUMNS,
)


def gaussian_kernel(window_size: int = 7, sigma: float | None = None) -> np.ndarray:
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("Gaussian window_size must be a positive odd integer")
    if window_size == 1 and sigma is None:
        return np.ones(1, dtype=np.float64)
    if sigma is None:
        # Reproduces the final Data/smoothing files: the selected experimental
        # window label 7 maps to sigma=3, with a four-sigma truncated kernel.
        sigma = (window_size - 1) / 2.0
    if sigma <= 0:
        raise ValueError("Gaussian sigma must be positive")
    radius = int(4.0 * sigma + 0.5)
    positions = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (positions / sigma) ** 2)
    return (kernel / kernel.sum()).astype(np.float64)


def gaussian_smooth(values: np.ndarray, window_size: int = 7, sigma: float | None = None) -> np.ndarray:
    """Smooth a [frames, features] array without SciPy, preserving its length."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D array, got shape={array.shape}")
    kernel = gaussian_kernel(window_size, sigma)
    radius = len(kernel) // 2
    padded = np.pad(array, ((radius, radius), (0, 0)), mode="edge")
    result = np.empty_like(array)
    for feature_idx in range(array.shape[1]):
        result[:, feature_idx] = np.convolve(
            padded[:, feature_idx], kernel, mode="valid"
        )
    return result


def _validate_binary_labels(frame: pd.DataFrame) -> None:
    for column in LABEL_COLUMNS:
        unique = set(frame[column].dropna().astype(int).unique().tolist())
        if not unique.issubset({0, 1}):
            raise ValueError(f"{column} must be binary, got {sorted(unique)}")


def preprocess_dataframe(
    frame: pd.DataFrame,
    *,
    gaussian_window: int = 7,
    gaussian_sigma: float | None = None,
    freeze_devices: bool = True,
) -> pd.DataFrame:
    """Return the canonical 16-node processed table used by all models."""

    work = frame.copy()
    work.columns = [str(column).lstrip("\ufeff") for column in work.columns]
    if "frame" not in work:
        raise ValueError("Input data must contain a frame column")

    missing_coordinates = [column for column in COORDINATE_COLUMNS if column not in work]
    missing_labels = [column for column in LABEL_COLUMNS if column not in work]
    if missing_coordinates:
        raise ValueError(f"Missing canonical coordinate columns: {missing_coordinates}")
    if missing_labels:
        # label_3 is a legacy ignore/transition flag; defaulting it to zero is safe.
        if missing_labels == ["label_3"]:
            work["label_3"] = 0
        else:
            raise ValueError(f"Missing label columns: {missing_labels}")

    work = work[["frame", *COORDINATE_COLUMNS, *LABEL_COLUMNS]].copy()
    work["frame"] = pd.to_numeric(work["frame"], errors="raise").astype(int)
    work = work.sort_values("frame").drop_duplicates("frame", keep="last").reset_index(drop=True)

    work[list(COORDINATE_COLUMNS)] = (
        work[list(COORDINATE_COLUMNS)]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .interpolate(limit_direction="both")
        .ffill()
        .bfill()
    )
    if work[list(COORDINATE_COLUMNS)].isna().any().any():
        missing = work[list(COORDINATE_COLUMNS)].columns[
            work[list(COORDINATE_COLUMNS)].isna().any()
        ].tolist()
        raise ValueError(f"Unable to interpolate coordinate columns: {missing}")

    body_values = work[list(BODY_COORDINATE_COLUMNS)].to_numpy(dtype=np.float64)
    work.loc[:, list(BODY_COORDINATE_COLUMNS)] = gaussian_smooth(
        body_values, window_size=gaussian_window, sigma=gaussian_sigma
    )

    if freeze_devices:
        for column in DEVICE_COORDINATE_COLUMNS:
            work[column] = float(work[column].median())

    for column in LABEL_COLUMNS:
        work[column] = pd.to_numeric(work[column], errors="coerce").fillna(0).astype(int)
    both_on = (work["label_device1"] == 1) & (work["label_device2"] == 1)
    work.loc[both_on, "label_3"] = 1
    _validate_binary_labels(work)
    return work


def preprocess_file(
    input_csv: str | Path,
    output_csv: str | Path,
    *,
    gaussian_window: int = 7,
    gaussian_sigma: float | None = None,
    freeze_devices: bool = True,
) -> Path:
    frame = pd.read_csv(input_csv, encoding="utf-8-sig")
    processed = preprocess_dataframe(
        frame,
        gaussian_window=gaussian_window,
        gaussian_sigma=gaussian_sigma,
        freeze_devices=freeze_devices,
    )
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output, index=False, encoding="utf-8-sig", float_format="%.9f")
    return output


def preprocess_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    pattern: str = "dataset_*.csv",
    gaussian_window: int = 7,
    gaussian_sigma: float | None = None,
) -> list[Path]:
    source = Path(input_dir)
    destination = Path(output_dir)
    files = sorted(source.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {source}")
    return [
        preprocess_file(
            path,
            destination / path.name,
            gaussian_window=gaussian_window,
            gaussian_sigma=gaussian_sigma,
        )
        for path in files
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Merged CSV or directory")
    parser.add_argument("--output", required=True, help="Processed CSV or directory")
    parser.add_argument("--window", type=int, default=7, help="Gaussian window (odd)")
    parser.add_argument("--sigma", type=float, default=None)
    parser.add_argument("--pattern", default="dataset_*.csv")
    parser.add_argument("--keep-device-track", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = Path(args.input)
    if source.is_dir():
        outputs = preprocess_directory(
            source,
            args.output,
            pattern=args.pattern,
            gaussian_window=args.window,
            gaussian_sigma=args.sigma,
        )
        for output in outputs:
            print(output)
    else:
        print(
            preprocess_file(
                source,
                args.output,
                gaussian_window=args.window,
                gaussian_sigma=args.sigma,
                freeze_devices=not args.keep_device_track,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
