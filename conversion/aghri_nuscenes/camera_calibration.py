"""AGHRI camera intrinsics and directed static-transform resolution."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .poses import Pose
from .transforms import make_transform, matrix_to_quaternion, normalize_quaternion


@dataclass(frozen=True)
class CameraCalibration:
    channel: str
    optical_frame: str
    width: int
    height: int
    camera_model: str
    distortion_model: str
    coefficient_order: tuple[str, ...]
    distortion_coefficients: np.ndarray
    intrinsic: np.ndarray
    rectification: np.ndarray
    projection: np.ndarray
    sensor_to_base: Pose
    directed_frame_path: tuple[str, ...]
    calibration_source: str
    image_state: str
    image_state_evidence: tuple[str, ...]


def _matrix(entry: dict, nested_key: str, flat_key: str, shape: tuple[int, int]) -> np.ndarray:
    if nested_key in entry:
        source = entry[nested_key]["data"]
    elif flat_key in entry:
        source = entry[flat_key]
    else:
        raise ValueError(f"missing calibration matrix {nested_key}/{flat_key}")
    value = np.asarray(source, dtype=np.float64)
    if value.size != shape[0] * shape[1] or not np.all(np.isfinite(value)):
        raise ValueError(f"invalid finite {shape} matrix for {nested_key}/{flat_key}")
    return value.reshape(shape)


def _vector(entry: dict, nested_key: str, flat_key: str) -> np.ndarray:
    if nested_key in entry:
        source = entry[nested_key]["data"]
    elif flat_key in entry:
        source = entry[flat_key]
    else:
        raise ValueError(f"missing calibration vector {nested_key}/{flat_key}")
    value = np.asarray(source, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"non-finite calibration vector {nested_key}/{flat_key}")
    return value


def _image_size(entry: dict) -> tuple[int, int]:
    width = entry.get("image_width", entry.get("width"))
    height = entry.get("image_height", entry.get("height"))
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError("invalid or missing calibration image size")
    return width, height


def _transform_record(record: dict) -> np.ndarray:
    translation = record["transform"]["translation"]
    rotation = record["transform"]["rotation"]
    return make_transform(
        [translation["x"], translation["y"], translation["z"]],
        [rotation["w"], rotation["x"], rotation["y"], rotation["z"]],
    )


def resolve_directed_transform(
    transforms: list[dict], parent_frame: str, child_frame: str
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Compose only declared parent-to-child edges into ``T_parent_child``."""
    adjacency: dict[str, list[tuple[str, np.ndarray]]] = {}
    for record in transforms:
        parent = record.get("header", {}).get("frame_id")
        child = record.get("child_frame_id")
        if not isinstance(parent, str) or not isinstance(child, str):
            raise ValueError("static transform record lacks named parent/child frames")
        adjacency.setdefault(parent, []).append((child, _transform_record(record)))

    queue: list[tuple[str, np.ndarray, tuple[str, ...]]] = [
        (parent_frame, np.eye(4, dtype=np.float64), (parent_frame,))
    ]
    solutions: list[tuple[np.ndarray, tuple[str, ...]]] = []
    while queue:
        frame, transform, path = queue.pop(0)
        if frame == child_frame:
            solutions.append((transform, path))
            continue
        for next_frame, edge in adjacency.get(frame, []):
            if next_frame not in path:
                queue.append((next_frame, transform @ edge, path + (next_frame,)))
    if len(solutions) != 1:
        raise ValueError(
            f"expected exactly one directed {parent_frame}->{child_frame} transform path; "
            f"found {len(solutions)}"
        )
    return solutions[0]


def load_camera_calibrations(
    intrinsics_path: Path,
    extrinsics_path: Path,
    definitions: list[dict[str, Any]],
) -> dict[str, CameraCalibration]:
    intrinsic_root = json.loads(intrinsics_path.read_text(encoding="utf-8"))
    extrinsic_root = json.loads(extrinsics_path.read_text(encoding="utf-8"))
    transforms = extrinsic_root.get("transforms")
    if not isinstance(intrinsic_root, dict) or not isinstance(transforms, list):
        raise ValueError("invalid camera calibration JSON roots")

    result: dict[str, CameraCalibration] = {}
    for definition in definitions:
        channel = definition["channel"]
        entry = intrinsic_root.get(channel)
        if not isinstance(entry, dict):
            raise ValueError(f"missing intrinsic entry for {channel}")
        optical_frame = definition["optical_frame"]
        if entry.get("header", {}).get("frame_id") != optical_frame:
            raise ValueError(
                f"{channel} intrinsic frame mismatch: "
                f"{entry.get('header', {}).get('frame_id')} != {optical_frame}"
            )
        distortion_model = entry.get("distortion_model")
        if distortion_model != definition["distortion_model"]:
            raise ValueError(
                f"{channel} distortion model mismatch: {distortion_model}"
            )
        intrinsic = _matrix(entry, "camera_matrix", "k", (3, 3))
        distortion = _vector(entry, "distortion_coefficients", "d")
        rectification = _matrix(entry, "rectification_matrix", "r", (3, 3))
        projection = _matrix(entry, "projection_matrix", "p", (3, 4))
        coefficient_order = tuple(definition["coefficient_order"])
        if distortion.size != len(coefficient_order):
            raise ValueError(
                f"{channel} has {distortion.size} coefficients but approved order has "
                f"{len(coefficient_order)} names"
            )
        width, height = _image_size(entry)
        if intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0:
            raise ValueError(f"{channel} focal lengths must be positive")
        if not (0 <= intrinsic[0, 2] < width and 0 <= intrinsic[1, 2] < height):
            raise ValueError(f"{channel} principal point is outside calibrated image")
        transform, path = resolve_directed_transform(transforms, "base_link", optical_frame)
        rotation = matrix_to_quaternion(transform[:3, :3])
        normalize_quaternion(rotation)
        result[channel] = CameraCalibration(
            channel=channel,
            optical_frame=optical_frame,
            width=width,
            height=height,
            camera_model=definition["camera_model"],
            distortion_model=distortion_model,
            coefficient_order=coefficient_order,
            distortion_coefficients=distortion,
            intrinsic=intrinsic,
            rectification=rectification,
            projection=projection,
            sensor_to_base=Pose(transform[:3, 3].copy(), rotation),
            directed_frame_path=path,
            calibration_source=definition["calibration_source"],
            image_state=definition["image_state"],
            image_state_evidence=tuple(definition["image_state_evidence"]),
        )
    if len(result) != len(definitions):
        raise ValueError("camera calibration definitions are not unique")
    return result
