"""Command dispatcher for the FYP package."""

from __future__ import annotations

import sys


COMMANDS = {
    "record-skeleton": (
        "source_code.data_collection.collect_skeleton",
        "Capture Azure Kinect skeleton and RGB",
    ),
    "locate-devices": (
        "device_detection.locate_devices_with_depth",
        "Calibrate device 3D positions",
    ),
    "train-device-detector": (
        "device_detection.train_device_detector",
        "Train the lamp/speaker YOLO segmentation model",
    ),
    "integrate": (
        "data_preparation.merge_data",
        "Merge skeleton, devices, and labels",
    ),
    "preprocess": (
        "data_preparation.smooth_data",
        "Interpolate and smooth canonical CSVs",
    ),
    "train": (
        "source_code.model_experiments.train_models",
        "Train XGBoost/CNN/BiLSTM/ST-GCN/final model",
    ),
    "realtime": (
        "real_time.run_realtime",
        "Replay real-time inference from a canonical CSV",
    ),
}


def _print_help() -> None:
    print("Usage: fyp <command> [options]\n")
    print("Commands:")
    width = max(len(command) for command in COMMANDS)
    for command, (_, description) in COMMANDS.items():
        print(f"  {command:<{width}}  {description}")
    print("\nRun 'fyp <command> --help' for command-specific options.")


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in ("-h", "--help", "help"):
        _print_help()
        return 0
    command = arguments.pop(0)
    if command not in COMMANDS:
        print(f"Unknown command: {command}\n", file=sys.stderr)
        _print_help()
        return 2
    module_name = COMMANDS[command][0]
    module = __import__(module_name, fromlist=["main"])
    return int(module.main(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
