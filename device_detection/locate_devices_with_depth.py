"""YOLO segmentation + aligned depth device-localization recorder."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


@dataclass
class Detection:
    label: str
    class_id: int
    confidence: float
    box: tuple[int, int, int, int]
    mask: np.ndarray
    position: np.ndarray | None = None


def point_cloud_median(
    depth_image: np.ndarray,
    detection: Detection,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_scale: float = 0.001,
) -> np.ndarray | None:
    """Estimate a robust 3D object center from its segmentation mask."""

    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required for mask erosion") from error
    x1, y1, x2, y2 = detection.box
    if x2 <= x1 or y2 <= y1:
        return None
    mask = detection.mask[y1:y2, x1:x2]
    depth = depth_image[y1:y2, x1:x2]
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    local_y, local_x = np.where(mask > 0)
    if len(local_y) < 10:
        return None
    z = depth[local_y, local_x].astype(np.float64) * depth_scale
    valid = (z > 0.1) & (z < 4.0)
    z, local_y, local_x = z[valid], local_y[valid], local_x[valid]
    if len(z) < 5:
        return None
    foreground = z < np.percentile(z, 20) + 0.15
    z, local_y, local_x = z[foreground], local_y[foreground], local_x[foreground]
    if not len(z):
        return None
    image_x = local_x + x1
    image_y = local_y + y1
    world_x = (image_x - cx) * z / fx
    world_y = (image_y - cy) * z / fy
    return np.asarray(
        (np.median(world_x), np.median(world_y), np.median(z)), dtype=np.float64
    )


def localize_device_session(
    model_path: str | Path,
    output_csv: str | Path,
    *,
    detection_interval: int = 10,
    confidence: float = 0.50,
) -> Path:
    """Run the original Azure Kinect device-calibration workflow."""

    try:
        import cv2
        from pyk4a import ColorResolution, Config, DepthMode, FPS, PyK4A
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError(
            "Device localization requires opencv-python, pyk4a, ultralytics, and "
            "the Azure Kinect Sensor SDK."
        ) from error

    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    camera = PyK4A(
        Config(
            color_resolution=ColorResolution.RES_720P,
            depth_mode=DepthMode.NFOV_UNBINNED,
            camera_fps=FPS.FPS_30,
            synchronized_images_only=True,
        )
    )
    camera.start()
    matrix = camera.calibration.get_camera_matrix(1)
    fx, fy, cx, cy = matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]
    detector = YOLO(str(model_path))
    detector(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
    tracked: dict[str, Detection] = {}
    frame_number = 0

    try:
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("Timestamp", "FrameID", "Label", "Conf", "X(m)", "Y(m)", "Z(m)"))
            while True:
                capture = camera.get_capture()
                if capture.color is None or capture.transformed_depth is None:
                    continue
                image = capture.color[:, :, :3].copy()
                depth = capture.transformed_depth
                if frame_number % detection_interval == 0:
                    result = detector(
                        image,
                        conf=confidence,
                        iou=0.5,
                        imgsz=640,
                        max_det=20,
                        verbose=False,
                        retina_masks=True,
                    )[0]
                    new: dict[str, Detection] = {}
                    masks = None if result.masks is None else result.masks.data.cpu().numpy()
                    if result.boxes is not None:
                        height, width = image.shape[:2]
                        for index, box in enumerate(result.boxes.cpu().numpy()):
                            class_id = int(box.cls[0])
                            label = str(result.names[class_id]).lower()
                            if label not in ("lamp", "speaker") or masks is None:
                                continue
                            x1, y1, x2, y2 = box.xyxy[0].astype(int)
                            x1, x2 = max(x1, 0), min(x2, width)
                            y1, y2 = max(y1, 0), min(y2, height)
                            mask = masks[index]
                            if mask.shape != (height, width):
                                mask = cv2.resize(mask, (width, height))
                            previous = tracked.get(label)
                            item = Detection(
                                label=label,
                                class_id=class_id,
                                confidence=float(box.conf[0]),
                                box=(x1, y1, x2, y2),
                                mask=(mask > 0.5).astype(np.uint8) * 255,
                                position=None if previous is None else previous.position,
                            )
                            new[label] = item
                    tracked = new

                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                for item in tracked.values():
                    raw = point_cloud_median(
                        depth,
                        item,
                        fx=fx,
                        fy=fy,
                        cx=cx,
                        cy=cy,
                    )
                    if raw is not None:
                        if item.position is None:
                            item.position = raw
                        else:
                            distance = np.linalg.norm(raw - item.position)
                            alpha = 0.02 if distance > 0.30 else 0.20
                            item.position = alpha * raw + (1.0 - alpha) * item.position
                    if item.position is None:
                        continue
                    x, y, z = item.position
                    writer.writerow(
                        (timestamp, frame_number, item.label, item.confidence, x, y, z)
                    )
                    x1, y1, x2, y2 = item.box
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        image,
                        f"{item.label} Z:{z:.3f}m",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )
                frame_number += 1
                cv2.imshow("Azure Kinect device localization", image)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    finally:
        camera.stop()
        cv2.destroyAllWindows()
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Fine-tuned YOLO segmentation weights")
    parser.add_argument("--output", required=True, help="Device-position CSV")
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--confidence", type=float, default=0.50)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(
        localize_device_session(
            args.model,
            args.output,
            detection_interval=args.interval,
            confidence=args.confidence,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
