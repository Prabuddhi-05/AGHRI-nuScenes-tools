"""Pose-stream loading, bounded interpolation and TF cross-checking."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .transforms import (
    invert_transform,
    make_transform,
    matrix_to_quaternion,
    normalize_quaternion,
    quaternion_angle,
    slerp,
)


@dataclass(frozen=True)
class Pose:
    translation: np.ndarray
    rotation_wxyz: np.ndarray

    @property
    def transform(self) -> np.ndarray:
        return make_transform(self.translation, self.rotation_wxyz)


@dataclass(frozen=True)
class PoseInterpolation:
    pose: Pose
    left_index: int
    right_index: int
    left_time_s: float
    right_time_s: float
    amount: float

    @property
    def bracket_ms(self) -> float:
        return (self.right_time_s - self.left_time_s) * 1_000.0


class PoseStream:
    def __init__(self, times_s: list[float], poses: list[Pose], label: str):
        if len(times_s) != len(poses) or not times_s:
            raise ValueError(f"empty or inconsistent pose stream: {label}")
        if any(b <= a for a, b in zip(times_s, times_s[1:])):
            raise ValueError(f"pose timestamps are not strictly increasing: {label}")
        self.times_s = times_s
        self.poses = poses
        self.label = label

    @property
    def time_range_s(self) -> tuple[float, float]:
        return self.times_s[0], self.times_s[-1]

    def interpolate(self, timestamp_ns: int, maximum_bracket_ms: float) -> PoseInterpolation:
        timestamp_s = timestamp_ns / 1_000_000_000.0
        index = bisect_left(self.times_s, timestamp_s)
        if index < len(self.times_s) and self.times_s[index] == timestamp_s:
            pose = self.poses[index]
            return PoseInterpolation(pose, index, index, timestamp_s, timestamp_s, 0.0)
        if index == 0 or index == len(self.times_s):
            raise ValueError(
                f"pose_extrapolation:{self.label}:{timestamp_s}:"
                f"{self.times_s[0]}:{self.times_s[-1]}"
            )
        left = index - 1
        right = index
        gap_s = self.times_s[right] - self.times_s[left]
        if gap_s * 1_000.0 > maximum_bracket_ms + 1e-9:
            raise ValueError(
                f"pose_bracket_exceeds_limit:{self.label}:{gap_s * 1000.0:.9f}ms"
            )
        amount = (timestamp_s - self.times_s[left]) / gap_s
        p0, p1 = self.poses[left], self.poses[right]
        translation = (1.0 - amount) * p0.translation + amount * p1.translation
        rotation = slerp(p0.rotation_wxyz, p1.rotation_wxyz, amount)
        return PoseInterpolation(
            Pose(translation, rotation), left, right,
            self.times_s[left], self.times_s[right], amount,
        )


def load_pose_stream(path: Path, translation_key: str, label: str) -> PoseStream:
    times: list[float] = []
    poses: list[Pose] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            translation_record = record[translation_key]
            rotation_record = record["q"]
            translation = np.array(
                [translation_record["x"], translation_record["y"], translation_record["z"]],
                dtype=np.float64,
            )
            rotation = normalize_quaternion(
                [rotation_record["w"], rotation_record["x"], rotation_record["y"], rotation_record["z"]]
            )
            if not np.all(np.isfinite(translation)):
                raise ValueError(f"non-finite pose translation at line {line_index}: {path}")
            times.append(float(record["t"]))
            poses.append(Pose(translation, rotation))
    return PoseStream(times, poses, label)


def rebase_pose(pose: Pose, first_pose: Pose) -> Pose:
    transform = invert_transform(first_pose.transform) @ pose.transform
    return Pose(transform[:3, 3].copy(), matrix_to_quaternion(transform[:3, :3]))


def compare_tf_composition(
    global_pose: Pose, map_odom_pose: Pose, odom_base_pose: Pose
) -> tuple[float, float]:
    composed = map_odom_pose.transform @ odom_base_pose.transform
    residual = invert_transform(global_pose.transform) @ composed
    translation_error = float(np.linalg.norm(residual[:3, 3]))
    rotation_error = quaternion_angle(matrix_to_quaternion(residual[:3, :3]))
    return translation_error, rotation_error


def resolve_lidar_calibration(extrinsics_path: Path, config: dict) -> tuple[Pose, dict]:
    with extrinsics_path.open("r", encoding="utf-8") as handle:
        root = json.load(handle)
    transforms = root.get("transforms")
    if not isinstance(transforms, list):
        raise ValueError("extrinsics.json must contain a transforms list")
    parent = config["required_parent"]
    child = config["required_child"]
    matches = [
        (index, record)
        for index, record in enumerate(transforms)
        if record.get("header", {}).get("frame_id") == parent
        and record.get("child_frame_id") == child
    ]
    if len(matches) != 1:
        raise ValueError(
            f"LiDAR calibration must resolve as one direct {parent}->{child} record; found {len(matches)}"
        )
    index, record = matches[0]
    tr = record["transform"]["translation"]
    qr = record["transform"]["rotation"]
    translation = np.array([tr["x"], tr["y"], tr["z"]], dtype=np.float64)
    rotation = normalize_quaternion([qr["w"], qr["x"], qr["y"], qr["z"]])
    tolerance = float(config["comparison_tolerance"])
    if not np.allclose(translation, config["expected_translation_m"], atol=tolerance, rtol=0):
        raise ValueError(f"LiDAR calibration translation changed: {translation.tolist()}")
    if not np.allclose(rotation, config["expected_rotation_wxyz"], atol=tolerance, rtol=0):
        raise ValueError(f"LiDAR calibration rotation changed: {rotation.tolist()}")
    evidence = {
        "record_index": index,
        "source_parent": parent,
        "source_child": child,
        "interpreted_transform": f"T_{parent}_{child}",
        "direction": f"maps points from {child} into {parent}",
        "translation_m": translation.tolist(),
        "rotation_wxyz": rotation.tolist(),
    }
    return Pose(translation, rotation), evidence
