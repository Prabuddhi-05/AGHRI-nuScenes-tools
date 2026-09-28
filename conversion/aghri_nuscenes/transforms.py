"""Quaternion and rigid-transform operations using [w, x, y, z]."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def normalize_quaternion(q: Iterable[float]) -> np.ndarray:
    value = np.asarray(tuple(q), dtype=np.float64)
    if value.shape != (4,) or not np.all(np.isfinite(value)):
        raise ValueError("quaternion must contain four finite numbers")
    norm = float(np.linalg.norm(value))
    if norm <= 0.0:
        raise ValueError("zero quaternion is invalid")
    return value / norm


def quaternion_multiply(left: Iterable[float], right: Iterable[float]) -> np.ndarray:
    w1, x1, y1, z1 = normalize_quaternion(left)
    w2, x2, y2, z2 = normalize_quaternion(right)
    return normalize_quaternion(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quaternion_inverse(q: Iterable[float]) -> np.ndarray:
    w, x, y, z = normalize_quaternion(q)
    return np.array([w, -x, -y, -z], dtype=np.float64)


def quaternion_to_matrix(q: Iterable[float]) -> np.ndarray:
    w, x, y, z = normalize_quaternion(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("rotation matrix must be 3x3")
    trace = float(np.trace(matrix))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = [0.25 * s, (matrix[2, 1] - matrix[1, 2]) / s,
             (matrix[0, 2] - matrix[2, 0]) / s, (matrix[1, 0] - matrix[0, 1]) / s]
    else:
        idx = int(np.argmax(np.diag(matrix)))
        if idx == 0:
            s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            q = [(matrix[2, 1] - matrix[1, 2]) / s, 0.25 * s,
                 (matrix[0, 1] + matrix[1, 0]) / s, (matrix[0, 2] + matrix[2, 0]) / s]
        elif idx == 1:
            s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            q = [(matrix[0, 2] - matrix[2, 0]) / s,
                 (matrix[0, 1] + matrix[1, 0]) / s, 0.25 * s,
                 (matrix[1, 2] + matrix[2, 1]) / s]
        else:
            s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            q = [(matrix[1, 0] - matrix[0, 1]) / s,
                 (matrix[0, 2] + matrix[2, 0]) / s,
                 (matrix[1, 2] + matrix[2, 1]) / s, 0.25 * s]
    result = normalize_quaternion(q)
    return -result if result[0] < 0 else result


def quaternion_angle(q: Iterable[float]) -> float:
    value = normalize_quaternion(q)
    return 2.0 * math.acos(min(1.0, abs(float(value[0]))))


def slerp(q0: Iterable[float], q1: Iterable[float], amount: float) -> np.ndarray:
    if not 0.0 <= amount <= 1.0:
        raise ValueError("SLERP amount must be in [0,1]")
    left = normalize_quaternion(q0)
    right = normalize_quaternion(q1)
    dot = float(np.dot(left, right))
    if dot < 0.0:
        right = -right
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return normalize_quaternion(left + amount * (right - left))
    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    return normalize_quaternion(
        math.sin((1.0 - amount) * theta_0) / sin_theta_0 * left
        + math.sin(amount * theta_0) / sin_theta_0 * right
    )


def make_transform(translation: Iterable[float], rotation_wxyz: Iterable[float]) -> np.ndarray:
    translation = np.asarray(tuple(translation), dtype=np.float64)
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError("translation must contain three finite numbers")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_to_matrix(rotation_wxyz)
    transform[:3, 3] = translation
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    rotation = transform[:3, :3]
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation.T
    result[:3, 3] = -rotation.T @ transform[:3, 3]
    return result


def transform_point(transform: np.ndarray, point: Iterable[float]) -> np.ndarray:
    point = np.asarray(tuple(point), dtype=np.float64)
    return np.asarray(transform, dtype=np.float64)[:3, :3] @ point + np.asarray(transform)[:3, 3]


def validate_rigid_transform(transform: np.ndarray, tolerance: float = 1e-9) -> None:
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("transform must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=tolerance):
        raise ValueError("invalid homogeneous transform last row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=tolerance):
        raise ValueError("rotation is not orthonormal")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=tolerance):
        raise ValueError("rotation determinant is not +1")
