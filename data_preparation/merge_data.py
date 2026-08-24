"""Turn raw skeleton, device-localization, and annotation files into one recording."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .schema import BODY_COORDINATE_COLUMNS, BODY_JOINT_NAMES, LABEL_COLUMNS, normalize_joint_name


def _find_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    lookup = {str(column).strip().lower(): str(column) for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise ValueError(f"Missing one of columns {candidates}; got {list(df.columns)}")


def _pick_primary_body(raw: pd.DataFrame, frame_col: str, body_col: str) -> pd.DataFrame:
    """Pick the most complete body in every frame, preferring the closest on ties."""

    work = raw.copy()
    work["_valid"] = (
        work[["_x", "_y", "_z"]].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        & (work["_z"].abs() > 1e-8)
    )
    quality = (
        work.groupby([frame_col, body_col], dropna=False)
        .agg(valid=("_valid", "sum"), median_z=("_z", "median"))
        .reset_index()
    )
    quality["median_z"] = quality["median_z"].fillna(np.inf)
    quality = quality.sort_values(
        [frame_col, "valid", "median_z"], ascending=[True, False, True]
    )
    chosen = quality.drop_duplicates(frame_col)[[frame_col, body_col]]
    return work.merge(chosen, on=[frame_col, body_col], how="inner")


def pivot_skeleton_csv(path: str | Path) -> pd.DataFrame:
    """Convert the recorder's long-form 32-joint CSV to 14-joint wide form."""

    raw = pd.read_csv(path, encoding="utf-8-sig")
    frame_col = _find_column(raw, ("frame number", "frame", "frameid", "frame_id"))
    body_col = _find_column(raw, ("body id", "body_id", "bodyid"))
    joint_col = _find_column(raw, ("joint name", "joint_name", "joint"))
    x_col = _find_column(raw, ("x", "x(m)"))
    y_col = _find_column(raw, ("y", "y(m)"))
    z_col = _find_column(raw, ("z", "z(m)"))

    raw = raw.rename(columns={x_col: "_x", y_col: "_y", z_col: "_z"})
    raw[frame_col] = pd.to_numeric(raw[frame_col], errors="raise").astype(int)
    for column in ("_x", "_y", "_z"):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    raw["_joint"] = raw[joint_col].map(normalize_joint_name)

    raw = _pick_primary_body(raw, frame_col, body_col)
    raw = raw[raw["_joint"].isin(BODY_JOINT_NAMES)].copy()
    if raw.empty:
        raise ValueError("No required upper-body joints were found in the skeleton CSV")

    long = raw.melt(
        id_vars=[frame_col, "_joint"],
        value_vars=["_x", "_y", "_z"],
        var_name="axis",
        value_name="value",
    )
    long["axis"] = long["axis"].str.removeprefix("_")
    long["feature"] = long["_joint"] + "_" + long["axis"]
    wide = long.pivot_table(index=frame_col, columns="feature", values="value", aggfunc="median")

    first_frame, last_frame = int(wide.index.min()), int(wide.index.max())
    wide = wide.reindex(range(first_frame, last_frame + 1))
    wide.index.name = "frame"
    wide = wide.reindex(columns=list(BODY_COORDINATE_COLUMNS))
    return wide.reset_index()


def load_device_positions(
    path: str | Path,
    frame_index: pd.Index,
    *,
    label_map: Mapping[str, str] | None = None,
    strategy: str = "median",
) -> pd.DataFrame:
    """Align YOLO/device coordinates with skeleton frames.

    ``median`` matches the final experiment: devices are calibrated once and remain
    fixed for the recording. ``interpolate`` keeps a time-varying device track.
    """

    raw = pd.read_csv(path, encoding="utf-8-sig")
    frame_col = _find_column(raw, ("frameid", "frame", "frame_id", "frame number"))
    label_col = _find_column(raw, ("label", "class", "name"))
    x_col = _find_column(raw, ("x(m)", "x"))
    y_col = _find_column(raw, ("y(m)", "y"))
    z_col = _find_column(raw, ("z(m)", "z"))

    mapping = {
        "lamp": "device1",
        "light": "device1",
        "device1": "device1",
        "speaker": "device2",
        "xiaoai": "device2",
        "device2": "device2",
    }
    if label_map:
        mapping.update({str(key).lower(): value for key, value in label_map.items()})

    work = pd.DataFrame(
        {
            "frame": pd.to_numeric(raw[frame_col], errors="coerce"),
            "device": raw[label_col].astype(str).str.strip().str.lower().map(mapping),
            "x": pd.to_numeric(raw[x_col], errors="coerce"),
            "y": pd.to_numeric(raw[y_col], errors="coerce"),
            "z": pd.to_numeric(raw[z_col], errors="coerce"),
        }
    ).dropna(subset=["frame", "device", "x", "y", "z"])
    work["frame"] = work["frame"].astype(int)

    missing_devices = {"device1", "device2"} - set(work["device"])
    if missing_devices:
        raise ValueError(f"Device log is missing detections for {sorted(missing_devices)}")

    if strategy == "median":
        stable = work.groupby("device")[["x", "y", "z"]].median()
        rows = pd.DataFrame(index=frame_index)
        for device in ("device1", "device2"):
            for axis in ("x", "y", "z"):
                rows[f"{device}_{axis}"] = float(stable.loc[device, axis])
        rows.index.name = "frame"
        return rows.reset_index()

    if strategy != "interpolate":
        raise ValueError("device strategy must be 'median' or 'interpolate'")

    long = work.melt(
        id_vars=["frame", "device"],
        value_vars=["x", "y", "z"],
        var_name="axis",
        value_name="value",
    )
    long["feature"] = long["device"] + "_" + long["axis"]
    wide = long.pivot_table(index="frame", columns="feature", values="value", aggfunc="median")
    wide = wide.reindex(frame_index).interpolate(limit_direction="both").ffill().bfill()
    wide.index.name = "frame"
    return wide.reset_index()


def _state_to_labels(value: object) -> tuple[int, int]:
    if isinstance(value, (int, np.integer)) or (isinstance(value, float) and value.is_integer()):
        state = int(value)
    else:
        normalized = str(value).strip().lower().replace(" ", "_")
        aliases = {
            "off": 0,
            "0": 0,
            "device1": 1,
            "on_device1": 1,
            "lamp": 1,
            "1": 1,
            "device2": 2,
            "on_device2": 2,
            "speaker": 2,
            "2": 2,
            "ignore": 3,
            "invalid": 3,
            "3": 3,
        }
        if normalized not in aliases:
            raise ValueError(f"Unknown attention state: {value!r}")
        state = aliases[normalized]
    if state not in (0, 1, 2, 3):
        raise ValueError(f"Attention state must be 0..3, got {state}")
    return (state & 1, (state >> 1) & 1)


def load_labels(path: str | Path, frame_index: pd.Index) -> pd.DataFrame:
    """Load frame labels or inclusive ``start_frame,end_frame,state`` intervals."""

    raw = pd.read_csv(path, encoding="utf-8-sig")
    lower = {str(column).strip().lower(): str(column) for column in raw.columns}
    labels = pd.DataFrame(index=frame_index, data={"label_device1": 0, "label_device2": 0, "label_3": 0})
    labels.index.name = "frame"

    if {"start_frame", "end_frame"}.issubset(lower):
        state_col = lower.get("state") or lower.get("label") or lower.get("attention_state")
        if state_col is None:
            raise ValueError("Interval annotations need a state/label column")
        for row in raw.itertuples(index=False):
            values = row._asdict()
            start = int(values[lower["start_frame"]])
            end = int(values[lower["end_frame"]])
            d1, d2 = _state_to_labels(values[state_col])
            mask = (labels.index >= start) & (labels.index <= end)
            labels.loc[mask, ["label_device1", "label_device2"]] = (d1, d2)
            if d1 == 1 and d2 == 1:
                labels.loc[mask, "label_3"] = 1
        return labels.reset_index()

    frame_col = lower.get("frame") or lower.get("frameid") or lower.get("frame number")
    if frame_col is None:
        raise ValueError("Frame annotations need a frame column or interval columns")

    if "state" in lower or "attention_state" in lower:
        state_col = lower.get("state") or lower["attention_state"]
        pairs = raw[state_col].map(_state_to_labels)
        frame_labels = pd.DataFrame(
            {
                "frame": pd.to_numeric(raw[frame_col], errors="raise").astype(int),
                "label_device1": [pair[0] for pair in pairs],
                "label_device2": [pair[1] for pair in pairs],
            }
        )
        frame_labels["label_3"] = (
            (frame_labels["label_device1"] == 1) & (frame_labels["label_device2"] == 1)
        ).astype(int)
    else:
        d1_col = lower.get("label_device1")
        d2_col = lower.get("label_device2")
        if d1_col is None or d2_col is None:
            raise ValueError("Frame annotations need state or label_device1/label_device2")
        frame_labels = pd.DataFrame(
            {
                "frame": pd.to_numeric(raw[frame_col], errors="raise").astype(int),
                "label_device1": pd.to_numeric(raw[d1_col], errors="raise").astype(int),
                "label_device2": pd.to_numeric(raw[d2_col], errors="raise").astype(int),
                "label_3": (
                    pd.to_numeric(raw[lower["label_3"]], errors="raise").astype(int)
                    if "label_3" in lower
                    else 0
                ),
            }
        )

    frame_labels = frame_labels.set_index("frame").reindex(frame_index).ffill().fillna(0).astype(int)
    return frame_labels.reset_index()


def integrate_recording(
    skeleton_csv: str | Path,
    device_csv: str | Path,
    labels_csv: str | Path,
    output_csv: str | Path,
    *,
    device_strategy: str = "median",
    skeleton_missing: str = "zero",
) -> Path:
    """Create one canonical wide CSV from the three raw sources."""

    skeleton = pivot_skeleton_csv(skeleton_csv)
    frame_index = pd.Index(skeleton["frame"].astype(int), name="frame")
    devices = load_device_positions(device_csv, frame_index, strategy=device_strategy)
    labels = load_labels(labels_csv, frame_index)

    merged = skeleton.merge(devices, on="frame", how="left").merge(labels, on="frame", how="left")
    coordinate_columns = [column for column in merged.columns if column.endswith(("_x", "_y", "_z"))]
    merged[coordinate_columns] = merged[coordinate_columns].replace([np.inf, -np.inf], np.nan)
    if skeleton_missing == "zero":
        # The final training CSVs preserve the recorder's zero-valued tracking-loss
        # convention before Gaussian smoothing. This is the checkpoint-compatible path.
        merged[coordinate_columns] = merged[coordinate_columns].fillna(0.0)
    elif skeleton_missing == "interpolate":
        merged[coordinate_columns] = merged[coordinate_columns].interpolate(
            limit_direction="both"
        )
    else:
        raise ValueError("skeleton_missing must be 'zero' or 'interpolate'")
    if merged[coordinate_columns].isna().any().any():
        missing = merged[coordinate_columns].columns[merged[coordinate_columns].isna().any()].tolist()
        raise ValueError(f"Coordinates still contain missing values after alignment: {missing}")

    for column in LABEL_COLUMNS:
        merged[column] = merged[column].fillna(0).astype(int)
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False, encoding="utf-8-sig")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skeleton", required=True, help="Long-form Azure skeleton CSV")
    parser.add_argument("--devices", required=True, help="YOLO 3D device-position CSV")
    parser.add_argument("--labels", required=True, help="Frame or interval attention labels")
    parser.add_argument("--output", required=True, help="Output merged CSV")
    parser.add_argument("--device-strategy", choices=("median", "interpolate"), default="median")
    parser.add_argument(
        "--skeleton-missing",
        choices=("zero", "interpolate"),
        default="zero",
        help="Keep checkpoint-compatible zeros or interpolate missing skeleton values",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = integrate_recording(
        args.skeleton,
        args.devices,
        args.labels,
        args.output,
        device_strategy=args.device_strategy,
        skeleton_missing=args.skeleton_missing,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
