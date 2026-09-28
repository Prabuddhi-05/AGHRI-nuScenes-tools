"""Constants for the AGHRI legacy-MMDetection3D/BEVFusion PKL variants."""

from __future__ import annotations

DATASET_VERSION = "v1.0-aghri"
SOURCE_DATASET = "AGHRI_nuScenes_camera_lidar_full"

LIDAR_CHANNEL = "lidar"
CAMERA_VARIANTS = {
    "full": (
        "cam_zed_rgb",
        "cam_fish_front",
        "cam_fish_left",
        "cam_fish_right",
    ),
    "zed_lidar": ("cam_zed_rgb",),
    "fisheye_lidar": (
        "cam_fish_front",
        "cam_fish_left",
        "cam_fish_right",
    ),
}
OUTPUT_PREFIXES = {
    "full": "aghri_full_infos",
    "zed_lidar": "aghri_zed_lidar_infos",
    "fisheye_lidar": "aghri_fisheye_lidar_infos",
}
# Backward-compatible alias used by helper modules that describe the original
# all-camera contract.
CAMERA_CHANNELS = CAMERA_VARIANTS["full"]
ACTIVE_CHANNELS = (LIDAR_CHANNEL, *CAMERA_CHANNELS)
DETECTION_CLASS = "human"
POINT_LOAD_DIM = 5
POINT_USE_DIM = 5
MAX_ALLOWED_SYNC_THRESHOLD_S = 0.50
MAX_VELOCITY_NEIGHBOUR_GAP_S = 1.0
PKL_PROTOCOL = 4

SPLIT_KEY_TO_NAME = {
    "aghri_train": "train",
    "aghri_val": "val",
    "aghri_test": "test",
}
INFO_REQUIRED_KEYS = (
    "lidar_path",
    "token",
    "sweeps",
    "cams",
    "lidar2ego_translation",
    "lidar2ego_rotation",
    "ego2global_translation",
    "ego2global_rotation",
    "timestamp",
    "gt_boxes",
    "gt_names",
    "gt_velocity",
    "num_lidar_pts",
    "num_radar_pts",
    "valid_flag",
    "num_features",
    "aghri_sync",
)

CAMERA_REQUIRED_KEYS = (
    "data_path",
    "type",
    "sample_data_token",
    "sensor2ego_translation",
    "sensor2ego_rotation",
    "ego2global_translation",
    "ego2global_rotation",
    "timestamp",
    "sensor2lidar_rotation",
    "sensor2lidar_translation",
    "cam_intrinsic",
    "camera_model_token",
    "camera_model",
    "distortion_model",
    "distortion_coefficients",
    "distortion_coefficient_order",
    "image_state",
    "image_width",
    "image_height",
    "calibration_source",
    "time_offset_s",
    "aligned_timestamp",
    "aligned_timestamp_ns",
    "sync_residual_s",
    "chosen_threshold_s",
)


def camera_channels(variant: str) -> tuple[str, ...]:
    """Return and validate the exact ordered camera channels for a variant."""
    try:
        return CAMERA_VARIANTS[variant]
    except KeyError as error:
        raise ValueError(f"unknown sensor configuration: {variant}") from error


def output_filename(variant: str, split: str) -> str:
    """Return the canonical PKL filename for a sensor configuration."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"unknown split: {split}")
    try:
        prefix = OUTPUT_PREFIXES[variant]
    except KeyError as error:
        raise ValueError(f"unknown sensor configuration: {variant}") from error
    return f"{prefix}_{split}.pkl"


def metadata(variant: str = "full") -> dict:
    """Return a fresh, insertion-ordered metadata dictionary."""
    return {
        "version": "v1.0-aghri",
        "dataset": "AGHRI",
        "classes": [DETECTION_CLASS],
        "sensor_configuration": variant,
        "camera_channels": list(camera_channels(variant)),
        "lidar_channel": LIDAR_CHANNEL,
        "point_cloud_load_dim": POINT_LOAD_DIM,
        "point_cloud_use_dim": POINT_USE_DIM,
        "point_feature_semantics": [
            "x",
            "y",
            "z",
            "placeholder_zero_not_measured_intensity",
            "placeholder_zero_not_measured_ring_or_time",
        ],
        "image_policy": "raw_distortion_ignored_pinhole_baseline",
        "detection_class": DETECTION_CLASS,
        "box_format": ["x", "y", "z", "x_size", "y_size", "z_size", "yaw"],
        "box_yaw_conversion": "target_yaw=-nuscenes_lidar_yaw-pi/2",
        "velocity_policy": (
            "same-instance finite differences in global coordinates, rotated into "
            "key LiDAR; NaN only when no neighbour within 1.0 s; audited loader maps "
            "that explicit missing value to zero before model use"
        ),
        "ground_truth_database_sampler": "disabled_no_database_created",
        "map_policy": "empty_no_map_pipeline",
        "source_dataset": SOURCE_DATASET,
        "source_dataset_version": DATASET_VERSION,
    }
