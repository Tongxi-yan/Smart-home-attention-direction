"""Shared dataset schema and Azure Kinect joint naming."""

from __future__ import annotations

from typing import Final

FPS: Final[int] = 30
WINDOW_SIZE: Final[int] = 20
WINDOW_STRIDE: Final[int] = 3
HISTORY_FRAMES: Final[int] = 5

CLASS_NAMES: Final[tuple[str, ...]] = ("off", "device1", "device2")

BODY_JOINT_NAMES: Final[tuple[str, ...]] = (
    "spine_navel",
    "neck",
    "head",
    "nose",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "left_hand",
    "left_handtip",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "right_hand",
    "right_handtip",
)
NODE_NAMES: Final[tuple[str, ...]] = BODY_JOINT_NAMES + ("device1", "device2")

COORDINATE_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"{node}_{axis}" for node in NODE_NAMES for axis in ("x", "y", "z")
)
BODY_COORDINATE_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"{node}_{axis}" for node in BODY_JOINT_NAMES for axis in ("x", "y", "z")
)
DEVICE_COORDINATE_COLUMNS: Final[tuple[str, ...]] = tuple(
    f"{node}_{axis}" for node in ("device1", "device2") for axis in ("x", "y", "z")
)

LABEL_COLUMNS: Final[tuple[str, ...]] = (
    "label_device1",
    "label_device2",
    "label_3",
)

# Official Azure Kinect Body Tracking SDK joint order (32 joints).
AZURE_JOINT_NAMES: Final[tuple[str, ...]] = (
    "pelvis",
    "spine_navel",
    "spine_chest",
    "neck",
    "left_clavicle",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "left_hand",
    "left_handtip",
    "left_thumb",
    "right_clavicle",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "right_hand",
    "right_handtip",
    "right_thumb",
    "left_hip",
    "left_knee",
    "left_ankle",
    "left_foot",
    "right_hip",
    "right_knee",
    "right_ankle",
    "right_foot",
    "head",
    "nose",
    "left_eye",
    "left_ear",
    "right_eye",
    "right_ear",
)

_ALIASES: Final[dict[str, str]] = {
    "spine - navel": "spine_navel",
    "spine navel": "spine_navel",
    "spine - chest": "spine_chest",
    "spine chest": "spine_chest",
    "left shoulder": "left_shoulder",
    "left elbow": "left_elbow",
    "left wrist": "left_wrist",
    "left hand": "left_hand",
    "left handtip": "left_handtip",
    "left hand tip": "left_handtip",
    "right shoulder": "right_shoulder",
    "right elbow": "right_elbow",
    "right wrist": "right_wrist",
    "right hand": "right_hand",
    "right handtip": "right_handtip",
    "right hand tip": "right_handtip",
}


def normalize_joint_name(name: str) -> str:
    """Convert raw SDK/display joint names to stable snake_case names."""

    normalized = " ".join(str(name).strip().lower().replace("_", " ").split())
    if normalized in _ALIASES:
        return _ALIASES[normalized]
    return normalized.replace(" - ", "_").replace("-", "_").replace(" ", "_")


def state_from_binary_labels(device1: int, device2: int) -> int:
    """Return 0=off, 1=device1, 2=device2, 3=invalid/both-on."""

    if device1 not in (0, 1) or device2 not in (0, 1):
        raise ValueError("device labels must be binary")
    return int(device1 + 2 * device2)
