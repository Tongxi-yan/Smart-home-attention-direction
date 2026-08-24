"""Azure Kinect skeleton and RGB recorder (Windows + NVIDIA runtime)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from data_preparation.schema import AZURE_JOINT_NAMES


def record_skeleton_session(
    output_directory: str | Path,
    *,
    gpu_id: int = 0,
    show_preview: bool = True,
) -> tuple[Path, Path]:
    """Record the long-form 32-joint CSV and synchronized color video.

    Hardware imports are local so the rest of the project remains usable on machines
    without the Azure Kinect SDK.
    """

    try:
        import cv2
        import pykinect_azure as pykinect
    except ImportError as error:
        raise RuntimeError(
            "Skeleton capture requires OpenCV, pykinect-azure, the Azure Kinect SDK, "
            "and the Body Tracking SDK on Windows."
        ) from error

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "body_joint_coordinates.csv"
    video_path = output / "color_video.mp4"

    pykinect.initialize_libraries(track_body=True)
    device_config = pykinect.default_configuration
    device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_UNBINNED
    device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30
    device = pykinect.start_device(config=device_config)

    tracker_config = pykinect.k4abt_tracker_configuration_t()
    tracker_config.sensor_orientation = pykinect.K4ABT_SENSOR_ORIENTATION_DEFAULT
    tracker_config.tracker_processing_mode = (
        pykinect.K4ABT_TRACKER_PROCESSING_MODE_GPU
    )
    tracker_config.gpu_device_id = gpu_id
    body_tracker = pykinect.start_body_tracker(tracker_config)
    video_writer = None
    frame_number = 0

    try:
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("Frame Number", "Body ID", "Joint Name", "X", "Y", "Z"))
            while True:
                capture = device.update()
                body_frame = body_tracker.update()
                color_ok, color_image = capture.get_color_image()
                if color_ok:
                    color_bgr = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
                    if video_writer is None:
                        height, width = color_bgr.shape[:2]
                        codec = cv2.VideoWriter_fourcc(*"mp4v")
                        video_writer = cv2.VideoWriter(
                            str(video_path), codec, 30.0, (width, height)
                        )
                    video_writer.write(color_bgr)

                bodies = body_frame.get_bodies()
                if bodies:
                    for body_index, body in enumerate(bodies):
                        body_id = getattr(body, "id", body_index)
                        for joint_index, joint in enumerate(body.joints):
                            name = (
                                AZURE_JOINT_NAMES[joint_index]
                                if joint_index < len(AZURE_JOINT_NAMES)
                                else f"joint_{joint_index}"
                            )
                            position = joint.position
                            writer.writerow(
                                (
                                    frame_number,
                                    body_id,
                                    name,
                                    position.x * 0.001,
                                    position.y * 0.001,
                                    position.z * 0.001,
                                )
                            )
                else:
                    for name in AZURE_JOINT_NAMES:
                        writer.writerow((frame_number, 0, name, 0.0, 0.0, 0.0))

                if show_preview:
                    depth_ok, depth_image = capture.get_colored_depth_image()
                    segment_ok, segment_image = body_frame.get_segmentation_image()
                    if depth_ok and segment_ok:
                        preview = cv2.addWeighted(depth_image, 0.6, segment_image, 0.4, 0)
                        preview = body_frame.draw_bodies(preview)
                        cv2.imshow("Azure Kinect skeleton recorder", preview)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                frame_number += 1
    finally:
        if video_writer is not None:
            video_writer.release()
        device.stop_device()
        body_tracker.stop_body_tracker()
        cv2.destroyAllWindows()
    return csv_path, video_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Recording directory")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--no-preview", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    csv_path, video_path = record_skeleton_session(
        args.output, gpu_id=args.gpu_id, show_preview=not args.no_preview
    )
    print(csv_path)
    print(video_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
