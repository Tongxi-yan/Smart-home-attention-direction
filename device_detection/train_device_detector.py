"""Two-stage training and evaluation for the lamp/speaker YOLO segmenter."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


DEFAULT_SEED = 3407


def _yolo_class():
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Device-detector training requires ultralytics: pip install -e '.[azure]'"
        ) from error
    return YOLO


def _best_weight(model: Any) -> Path:
    run_directory = Path(model.trainer.save_dir)
    best = run_directory / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Best model file not found: {best}")
    return best


def train_stage_a(
    data_yaml: str | Path,
    output_directory: str | Path,
    *,
    base_model: str = "yolov8s-seg.pt",
    device: str = "0",
    seed: int = DEFAULT_SEED,
) -> Path:
    """Base training with the final 200-epoch augmentation schedule."""

    print("\n========== Starting Stage A (Base Training) ==========\n")
    model = _yolo_class()(base_model)
    model.train(
        data=str(data_yaml),
        task="segment",
        device=device,
        project=str(output_directory),
        name="stage_a",
        imgsz=640,
        batch=32,
        workers=16,
        epochs=200,
        patience=30,
        cache=True,
        seed=seed,
        optimizer="SGD",
        lr0=0.001,
        momentum=0.937,
        weight_decay=1e-4,
        cos_lr=True,
        warmup_epochs=3,
        mosaic=1.0,
        close_mosaic=10,
        hsv_h=0.01,
        hsv_s=0.35,
        hsv_v=0.25,
        amp=True,
        plots=True,
    )
    best = _best_weight(model)
    print(f"Stage A complete. Best checkpoint: {best}")
    return best


def train_stage_b(
    finetune_from: str | Path,
    data_yaml: str | Path,
    output_directory: str | Path,
    *,
    device: str = "0",
    seed: int = DEFAULT_SEED,
) -> Path:
    """High-resolution fine-tuning with weak augmentation."""

    print("\n========== Starting Stage B (Fine-tuning) ==========\n")
    model = _yolo_class()(str(finetune_from))
    model.train(
        data=str(data_yaml),
        task="segment",
        device=device,
        project=str(output_directory),
        name="stage_b",
        imgsz=768,
        batch=16,
        workers=16,
        epochs=100,
        patience=15,
        cache=True,
        seed=seed,
        optimizer="SGD",
        lr0=0.0005,
        momentum=0.937,
        weight_decay=1e-4,
        cos_lr=True,
        warmup_epochs=1,
        mosaic=0.0,
        mixup=0.0,
        hsv_h=0.005,
        hsv_s=0.20,
        hsv_v=0.15,
        amp=True,
        plots=True,
    )
    best = _best_weight(model)
    print(f"Stage B complete. Best checkpoint: {best}")
    return best


def evaluate_detector(
    checkpoint: str | Path,
    data_yaml: str | Path,
    *,
    device: str = "0",
) -> None:
    """Report segmentation mAP for validation and test partitions."""

    model = _yolo_class()(str(checkpoint))

    def print_report(metrics: Any, split_name: str) -> None:
        print(f"\n[{split_name}] segmentation metrics")
        print(f"mAP@50:    {metrics.seg.map50:.2%}")
        print(f"mAP@50-95: {metrics.seg.map:.2%}")
        print("Per-class mAP@50-95:")
        for index, class_name in model.names.items():
            print(f"  {class_name:<10}: {metrics.seg.maps[index]:.2%}")

    validation = model.val(
        data=str(data_yaml), split="val", device=device, imgsz=768, verbose=False
    )
    print_report(validation, "Validation")
    try:
        test = model.val(
            data=str(data_yaml), split="test", device=device, imgsz=768, verbose=False
        )
    except Exception as error:
        print(f"Test evaluation unavailable: {error}")
    else:
        print_report(test, "Test")


def train_detector(
    data_yaml: str | Path,
    output_directory: str | Path,
    *,
    base_model: str = "yolov8s-seg.pt",
    device: str = "0",
    seed: int = DEFAULT_SEED,
    evaluate: bool = True,
) -> Path:
    """Run both preserved training stages and optionally evaluate the result."""

    stage_a = train_stage_a(
        data_yaml,
        output_directory,
        base_model=base_model,
        device=device,
        seed=seed,
    )
    stage_b = train_stage_b(
        stage_a,
        data_yaml,
        output_directory,
        device=device,
        seed=seed,
    )
    if evaluate:
        evaluate_detector(stage_b, data_yaml, device=device)
    return stage_b


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="device_detection/detector_config.yaml")
    parser.add_argument("--output", default="outputs/device_detection")
    parser.add_argument("--base-model", default="yolov8s-seg.pt")
    parser.add_argument("--device", default="0", help="CUDA index such as 0, or cpu")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--skip-evaluation", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    best = train_detector(
        args.data,
        args.output,
        base_model=args.base_model,
        device=args.device,
        seed=args.seed,
        evaluate=not args.skip_evaluation,
    )
    print(f"Final device-detector checkpoint: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
