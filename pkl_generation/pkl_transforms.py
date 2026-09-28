"""Small, dependency-light rigid-transform helpers using nuScenes wxyz quaternions."""

from __future__ import annotations

import math
import numpy as np


def normalize_quaternion(q) -> np.ndarray:
    value = np.asarray(q, dtype=np.float64)
    if value.shape != (4,) or not np.all(np.isfinite(value)):
        raise ValueError(f"invalid quaternion: {q!r}")
    norm = float(np.linalg.norm(value))
    if norm <= 0.0:
        raise ValueError("zero quaternion")
    return value / norm


def quaternion_to_matrix(q) -> np.ndarray:
    w, x, y, z = normalize_quaternion(q)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion(matrix) -> np.ndarray:
    """Convert a proper rotation matrix to canonical-sign wxyz quaternion."""
    m = np.asarray(matrix, dtype=np.float64)
    validate_rotation(m)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.asarray(
            [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
             (m[1, 0] - m[0, 1]) / s]
        )
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = np.asarray(
            [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s,
             (m[0, 2] + m[2, 0]) / s]
        )
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = np.asarray(
            [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s,
             (m[1, 2] + m[2, 1]) / s]
        )
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = np.asarray(
            [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
             (m[1, 2] + m[2, 1]) / s, 0.25 * s]
        )
    q = normalize_quaternion(q)
    return -q if q[0] < 0 else q


def make_transform(translation, rotation_wxyz) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_to_matrix(rotation_wxyz)
    value = np.asarray(translation, dtype=np.float64)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"invalid translation: {translation!r}")
    transform[:3, 3] = value
    return transform


def invert_transform(transform) -> np.ndarray:
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4):
        raise ValueError("rigid transform must be 4x4")
    validate_rotation(value[:3, :3])
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = value[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ value[:3, 3]
    return result


def transform_points(transform, points) -> np.ndarray:
    value = np.asarray(points, dtype=np.float64)
    return (np.asarray(transform)[:3, :3] @ value.T).T + np.asarray(transform)[:3, 3]


def validate_rotation(rotation, atol: float = 1e-8) -> None:
    value = np.asarray(rotation, dtype=np.float64)
    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    if not np.allclose(value.T @ value, np.eye(3), atol=atol):
        raise ValueError("rotation matrix is not orthonormal")
    if not math.isclose(float(np.linalg.det(value)), 1.0, abs_tol=atol):
        raise ValueError("rotation determinant is not +1")


def yaw_from_matrix(rotation) -> float:
    value = np.asarray(rotation, dtype=np.float64)
    return float(math.atan2(value[1, 0], value[0, 0]))


def yaw_matrix(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotation_distance_rad(left, right) -> float:
    relative = np.asarray(left).T @ np.asarray(right)
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(math.acos(cosine))

