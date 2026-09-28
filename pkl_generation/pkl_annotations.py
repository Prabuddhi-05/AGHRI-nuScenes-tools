"""Convert global annotations into legacy MMDetection3D LiDAR boxes."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import numpy as np

from pkl_contract import MAX_VELOCITY_NEIGHBOUR_GAP_S
from pkl_transforms import quaternion_to_matrix, transform_points, yaw_from_matrix


@dataclass(frozen=True)
class VelocityEstimate:
    global_xy: np.ndarray
    method: str
    span_s: float | None


def build_annotation_tracks(
    annotations: list[dict], sample_by_token: dict[str, dict]
) -> dict[str, list[tuple[int, np.ndarray, str]]]:
    tracks = defaultdict(list)
    for annotation in annotations:
        sample = sample_by_token[annotation["sample_token"]]
        tracks[annotation["instance_token"]].append(
            (
                int(sample["timestamp"]),
                np.asarray(annotation["translation"], dtype=np.float64),
                annotation["token"],
            )
        )
    for values in tracks.values():
        values.sort(key=lambda item: (item[0], item[2]))
    return dict(tracks)


def estimate_track_velocities(
    tracks: dict[str, list[tuple[int, np.ndarray, str]]],
    maximum_gap_s: float = MAX_VELOCITY_NEIGHBOUR_GAP_S,
) -> dict[str, VelocityEstimate]:
    result: dict[str, VelocityEstimate] = {}
    max_gap_us = maximum_gap_s * 1_000_000.0
    for values in tracks.values():
        for index, (timestamp, position, token) in enumerate(values):
            previous = values[index - 1] if index > 0 else None
            following = values[index + 1] if index + 1 < len(values) else None
            previous_ok = previous is not None and timestamp - previous[0] <= max_gap_us
            following_ok = following is not None and following[0] - timestamp <= max_gap_us
            if previous_ok and following_ok:
                dt_s = (following[0] - previous[0]) / 1_000_000.0
                velocity = (following[1] - previous[1]) / dt_s
                method = "central_difference"
            elif following_ok:
                dt_s = (following[0] - timestamp) / 1_000_000.0
                velocity = (following[1] - position) / dt_s
                method = "forward_one_sided"
            elif previous_ok:
                dt_s = (timestamp - previous[0]) / 1_000_000.0
                velocity = (position - previous[1]) / dt_s
                method = "backward_one_sided"
            else:
                result[token] = VelocityEstimate(
                    np.asarray([math.nan, math.nan]), "missing_no_neighbour_within_1s", None
                )
                continue
            result[token] = VelocityEstimate(
                np.asarray(velocity[:2], dtype=np.float64), method, float(dt_s)
            )
    return result


def annotation_to_target_box(annotation: dict, lidar_from_global: np.ndarray) -> np.ndarray:
    center = transform_points(
        lidar_from_global, np.asarray(annotation["translation"], dtype=np.float64)[None]
    )[0]
    global_rotation = quaternion_to_matrix(annotation["rotation"])
    lidar_rotation = lidar_from_global[:3, :3] @ global_rotation
    nu_lidar_yaw = yaw_from_matrix(lidar_rotation)
    target_yaw = -nu_lidar_yaw - math.pi / 2.0
    width, length, height = map(float, annotation["size"])
    return np.asarray(
        [center[0], center[1], center[2], width, length, height, target_yaw],
        dtype=np.float64,
    )


def velocity_to_lidar(
    estimate: VelocityEstimate, lidar_from_global: np.ndarray
) -> np.ndarray:
    if not np.all(np.isfinite(estimate.global_xy)):
        return np.asarray([math.nan, math.nan], dtype=np.float64)
    global_vector = np.asarray([estimate.global_xy[0], estimate.global_xy[1], 0.0])
    return (lidar_from_global[:3, :3] @ global_vector)[:2].astype(np.float64)


def build_sample_annotation_arrays(
    annotations: list[dict],
    lidar_from_global: np.ndarray,
    velocity_by_annotation: dict[str, VelocityEstimate],
) -> tuple[dict[str, np.ndarray], list[dict]]:
    boxes = []
    velocities = []
    velocity_rows = []
    for annotation in annotations:
        estimate = velocity_by_annotation[annotation["token"]]
        boxes.append(annotation_to_target_box(annotation, lidar_from_global))
        velocity_lidar = velocity_to_lidar(estimate, lidar_from_global)
        velocities.append(velocity_lidar)
        velocity_rows.append(
            {
                "annotation_token": annotation["token"],
                "instance_token": annotation["instance_token"],
                "method": estimate.method,
                "finite_estimate": bool(np.all(np.isfinite(velocity_lidar))),
                "difference_span_s": estimate.span_s,
                "global_velocity_xy": (
                    estimate.global_xy.tolist()
                    if np.all(np.isfinite(estimate.global_xy)) else None
                ),
                "lidar_velocity_xy": (
                    velocity_lidar.tolist()
                    if np.all(np.isfinite(velocity_lidar)) else None
                ),
            }
        )
    count = len(annotations)
    arrays = {
        "gt_boxes": np.asarray(boxes, dtype=np.float64).reshape(count, 7),
        "gt_names": np.asarray(["human"] * count),
        "gt_velocity": np.asarray(velocities, dtype=np.float64).reshape(count, 2),
        "num_lidar_pts": np.asarray(
            [item["num_lidar_pts"] for item in annotations], dtype=np.int64
        ),
        "num_radar_pts": np.asarray(
            [item["num_radar_pts"] for item in annotations], dtype=np.int64
        ),
    }
    arrays["valid_flag"] = arrays["num_lidar_pts"] > 0
    return arrays, velocity_rows
