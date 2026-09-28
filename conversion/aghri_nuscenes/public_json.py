"""Canonical field ordering for public AGHRI nuScenes-style JSON tables."""

from __future__ import annotations

from typing import Any


PUBLIC_TABLE_FIELD_ORDER: dict[str, tuple[str, ...]] = {
    "category": ("token", "name", "description"),
    "sensor": ("token", "channel", "modality"),
    "calibrated_sensor": (
        "token", "sensor_token", "translation", "rotation", "camera_intrinsic",
    ),
    "ego_pose": ("token", "timestamp", "rotation", "translation"),
    "log": ("token", "logfile", "vehicle", "date_captured", "location"),
    "scene": (
        "token", "log_token", "nbr_samples", "first_sample_token",
        "last_sample_token", "name", "description",
    ),
    "sample": ("token", "timestamp", "prev", "next", "scene_token"),
    "sample_data": (
        "token", "sample_token", "ego_pose_token", "calibrated_sensor_token",
        "timestamp", "fileformat", "is_key_frame", "height", "width",
        "filename", "prev", "next",
    ),
    "instance": (
        "token", "category_token", "nbr_annotations",
        "first_annotation_token", "last_annotation_token",
    ),
    "sample_annotation": (
        "token", "sample_token", "instance_token", "visibility_token",
        "attribute_tokens", "translation", "size", "rotation", "prev",
        "next", "num_lidar_pts", "num_radar_pts",
    ),
    "camera_model": (
        "token", "calibrated_sensor_token", "channel", "image_width",
        "image_height", "camera_model", "distortion_model",
        "distortion_coefficients", "coefficient_order", "image_state",
        "rectification_matrix", "projection_matrix", "calibration_source",
    ),
}

EMPTY_STAGE1_TABLES = frozenset({"attribute", "visibility", "map"})
SPLIT_KEY_ORDER = ("aghri_train", "aghri_val", "aghri_test")
PUBLIC_TABLE_NAMES = frozenset(PUBLIC_TABLE_FIELD_ORDER) | EMPTY_STAGE1_TABLES | {"splits"}


def _check_record_keys(table_name: str, record_index: int, record: Any) -> tuple[str, ...]:
    if not isinstance(record, dict):
        raise ValueError(f"{table_name}[{record_index}] must be an object")
    expected = PUBLIC_TABLE_FIELD_ORDER[table_name]
    actual = tuple(record)
    missing = [key for key in expected if key not in record]
    extra = [key for key in actual if key not in expected]
    if missing or extra:
        raise ValueError(
            f"{table_name}[{record_index}] field mismatch: missing={missing}, extra={extra}"
        )
    return actual


def order_public_table(table_name: str, value: Any) -> Any:
    """Return a value with the exact public table and record key ordering."""
    if table_name not in PUBLIC_TABLE_NAMES:
        raise ValueError(f"unknown public table: {table_name}")
    if table_name == "splits":
        if not isinstance(value, dict):
            raise ValueError("splits must be an object")
        missing = [key for key in SPLIT_KEY_ORDER if key not in value]
        extra = [key for key in value if key not in SPLIT_KEY_ORDER]
        if missing or extra:
            raise ValueError(f"splits field mismatch: missing={missing}, extra={extra}")
        return {key: value[key] for key in SPLIT_KEY_ORDER}
    if not isinstance(value, list):
        raise ValueError(f"{table_name} must be a list")
    if table_name in EMPTY_STAGE1_TABLES:
        if value:
            raise ValueError(
                f"{table_name} is defined as empty for Stage 1; no non-empty field order is approved"
            )
        return []
    expected = PUBLIC_TABLE_FIELD_ORDER[table_name]
    ordered = []
    for index, record in enumerate(value):
        _check_record_keys(table_name, index, record)
        ordered.append({key: record[key] for key in expected})
    return ordered


def validate_public_table_order(table_name: str, value: Any) -> None:
    """Raise if parsed JSON does not retain the exact approved key ordering."""
    ordered = order_public_table(table_name, value)
    if table_name == "splits":
        if tuple(value) != SPLIT_KEY_ORDER:
            raise ValueError(
                f"splits key order mismatch: {tuple(value)} != {SPLIT_KEY_ORDER}"
            )
        return
    if table_name in EMPTY_STAGE1_TABLES:
        return
    for index, (actual, expected) in enumerate(zip(value, ordered)):
        if tuple(actual) != tuple(expected):
            raise ValueError(
                f"{table_name}[{index}] key order mismatch: "
                f"{tuple(actual)} != {tuple(expected)}"
            )
