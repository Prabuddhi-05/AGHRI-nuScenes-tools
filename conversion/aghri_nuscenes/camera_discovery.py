"""Read-only camera, calibration, timestamp and pose discovery."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .camera_calibration import CameraCalibration, load_camera_calibrations
from .camera_sync import TimedFile, estimate_constant_offset_ns, scan_timed_files
from .poses import compare_tf_composition, load_pose_stream


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _mode_encoding(mode: str) -> tuple[str, int]:
    mapping = {
        "RGB": ("rgb8", 3),
        "RGBA": ("rgba8", 4),
        "L": ("mono8", 1),
    }
    if mode not in mapping:
        raise ValueError(f"unsupported source image mode: {mode}")
    return mapping[mode]


def _inspect_images(files: list[TimedFile]) -> dict:
    observations: Counter[tuple[int, int, str, str, int]] = Counter()
    unreadable: list[dict] = []
    opaque_alpha = 0
    nonopaque_alpha = 0
    for item in files:
        try:
            with Image.open(item.path) as image:
                image.load()
                encoding, channels = _mode_encoding(image.mode)
                observations[(image.width, image.height, image.mode, encoding, channels)] += 1
                if image.mode == "RGBA":
                    alpha = np.asarray(image.getchannel("A"))
                    if bool(np.all(alpha == 255)):
                        opaque_alpha += 1
                    else:
                        nonopaque_alpha += 1
        except Exception as error:
            unreadable.append({
                "filename": item.name,
                "error_type": type(error).__name__,
                "error": str(error),
            })
    return {
        "decoded_files": len(files) - len(unreadable),
        "unreadable_files": unreadable,
        "observed_image_layouts": [
            {
                "width": key[0],
                "height": key[1],
                "pil_mode": key[2],
                "encoding": key[3],
                "channel_count": key[4],
                "file_count": count,
            }
            for key, count in sorted(observations.items())
        ],
        "rgba_alpha": {
            "fully_opaque_files": opaque_alpha,
            "files_with_nonopaque_values": nonopaque_alpha,
        },
    }


def _table_root(base_pilot: Path, input_version: str) -> Path:
    version_root = base_pilot / input_version
    if not version_root.is_dir():
        raise ValueError(f"missing input pilot version directory: {version_root}")
    return version_root


def discover_camera_inputs(
    source_root: Path,
    base_pilot: Path,
    scenario: str,
    config: dict,
) -> tuple[dict, dict]:
    """Inspect every camera input without writing to the source tree."""
    version_root = _table_root(base_pilot, config["input_version"])
    samples = _load_json(version_root / "sample.json")
    sample_data = _load_json(version_root / "sample_data.json")
    scenes = _load_json(version_root / "scene.json")
    instances = _load_json(version_root / "instance.json")
    mapping = _load_json(base_pilot / "manifests" / "source_to_scene.json")
    if len(scenes) != 1 or not samples:
        raise ValueError("camera conversion requires one non-empty staged LiDAR scene")
    mapped = [entry for entry in mapping["scenes"] if entry["source_name"] == scenario]
    if len(mapped) != 1 or mapped[0]["public_scene_name"] != scenes[0]["name"]:
        raise ValueError("scenario does not resolve to the input pilot's single public scene")
    scene_path = source_root / mapped[0]["source_relative_path"]
    if not scene_path.is_dir():
        raise ValueError(f"missing source recording: {scene_path}")

    all_lidar_files, lidar_unparseable = scan_timed_files(
        scene_path / "sensor_data" / "lidar", [".pcd"]
    )
    lidar_records = [record for record in sample_data if record["filename"].startswith("samples/lidar/")]
    available_lidar = {item.name: item for item in all_lidar_files}
    referenced_names = [
        Path(record["filename"]).name.split("__", 1)[-1].removesuffix(".bin")
        for record in lidar_records
    ]
    missing_lidar = [name for name in referenced_names if name not in available_lidar]
    lidar_files = [available_lidar[name] for name in referenced_names if name in available_lidar]
    if missing_lidar or len(lidar_files) != len(samples) or len(lidar_records) != len(samples):
        raise ValueError(
            f"LiDAR anchor closure failed: included_source={len(lidar_files)}, "
            f"input_pilot={len(lidar_records)}, samples={len(samples)}, missing={missing_lidar}"
        )
    lidar_ns = np.asarray([item.timestamp_ns for item in lidar_files], dtype=np.int64)

    intrinsics_path = source_root / config["calibration"]["intrinsics"]
    extrinsics_path = source_root / config["calibration"]["extrinsics"]
    calibrations = load_camera_calibrations(
        intrinsics_path, extrinsics_path, config["camera_channels"]
    )
    global_calibration_files = sorted(
        path.relative_to(source_root).as_posix()
        for path in (intrinsics_path, extrinsics_path)
        if path.is_file()
    )

    pose_cfg = config["pose"]
    pose_stream = load_pose_stream(scene_path / pose_cfg["source"], "p", "odom_global")
    map_odom_stream = load_pose_stream(
        scene_path / pose_cfg["map_odom_tf"], "tr", "map_to_odom"
    )
    odom_base_stream = load_pose_stream(
        scene_path / pose_cfg["odom_base_tf"], "tr", "odom_to_base_link"
    )
    camera_files: dict[str, list[TimedFile]] = {}
    camera_pose_exclusions: dict[str, dict[str, str]] = {}
    stream_reports: dict[str, dict] = {}
    for definition in config["camera_channels"]:
        channel = definition["channel"]
        directory = scene_path / definition["source_path"]
        files, unparseable = scan_timed_files(directory, definition["extensions"])
        if not directory.is_dir() or not files:
            raise ValueError(f"missing or empty camera directory for {channel}: {directory}")
        camera_files[channel] = files
        timestamps = [item.timestamp_ns for item in files]
        duplicates = sorted(
            timestamp for timestamp, count in Counter(timestamps).items() if count > 1
        )
        filename_duplicates = sorted(
            name for name, count in Counter(item.name for item in files).items() if count > 1
        )
        supported_paths = {item.path for item in files}
        other_files = sorted(
            path.name for path in directory.iterdir()
            if path.is_file() and path not in supported_paths
        )
        image_report = _inspect_images(files)
        pose_errors: list[dict] = []
        pose_brackets: list[float] = []
        translation_errors: list[float] = []
        rotation_errors: list[float] = []
        channel_pose_exclusions: dict[str, str] = {}
        for item in files:
            try:
                global_interpolation = pose_stream.interpolate(
                    item.timestamp_ns, float(pose_cfg["maximum_bracket_ms"])
                )
                map_odom_interpolation = map_odom_stream.interpolate(
                    item.timestamp_ns, float(pose_cfg["maximum_bracket_ms"])
                )
                odom_base_interpolation = odom_base_stream.interpolate(
                    item.timestamp_ns, float(pose_cfg["maximum_bracket_ms"])
                )
                translation_error, rotation_error = compare_tf_composition(
                    global_interpolation.pose,
                    map_odom_interpolation.pose,
                    odom_base_interpolation.pose,
                )
                if translation_error > float(
                    pose_cfg["tf_composition_translation_tolerance_m"]
                ) or rotation_error > float(
                    pose_cfg["tf_composition_rotation_tolerance_rad"]
                ):
                    reason = "tf_composition_tolerance_exceeded"
                    channel_pose_exclusions[item.name] = reason
                    pose_errors.append({
                        "filename": item.name,
                        "reason": reason,
                        "translation_error_m": translation_error,
                        "rotation_error_rad": rotation_error,
                    })
                    continue
                pose_brackets.append(global_interpolation.bracket_ms)
                translation_errors.append(translation_error)
                rotation_errors.append(rotation_error)
            except ValueError as error:
                reason = "pose_interpolation_or_tf_crosscheck_failed"
                channel_pose_exclusions[item.name] = reason
                pose_errors.append({
                    "filename": item.name,
                    "reason": reason,
                    "error": str(error),
                })
        camera_pose_exclusions[channel] = channel_pose_exclusions
        offset_ns, estimator_differences = estimate_constant_offset_ns(
            lidar_ns, np.asarray(timestamps, dtype=np.int64)
        )
        calibration = calibrations[channel]
        observed_layouts = image_report["observed_image_layouts"]
        calibration_matches_images = (
            len(observed_layouts) == 1
            and observed_layouts[0]["width"] == calibration.width
            and observed_layouts[0]["height"] == calibration.height
        )
        stream_reports[channel] = {
            "source_directory": str(directory),
            "preferred_directory_used": True,
            "file_count": len(files),
            "extensions": dict(sorted(Counter(item.extension for item in files).items())),
            "filename_pattern": "seconds_nanoseconds.extension with exactly nine nanosecond digits",
            "timestamp_unit": "integer nanoseconds derived from filename",
            "first_timestamp_ns": timestamps[0],
            "last_timestamp_ns": timestamps[-1],
            "duplicate_filenames": filename_duplicates,
            "duplicate_timestamps_ns": duplicates,
            "unparseable_supported_extension_filenames": unparseable,
            "other_files_in_directory": other_files,
            **image_report,
            "pose_valid_files": len(files) - len(pose_errors),
            "pose_invalid_files": pose_errors,
            "pose_invalid_policy": (
                "Exclude the individual measurement before synchronization/keyframe "
                "selection; do not fabricate or extrapolate a pose."
            ),
            "maximum_pose_bracket_ms": max(pose_brackets) if pose_brackets else None,
            "maximum_safe_tf_composition_translation_error_m": (
                max(translation_errors) if translation_errors else None
            ),
            "maximum_safe_tf_composition_rotation_error_rad": (
                max(rotation_errors) if rotation_errors else None
            ),
            "calibration_resolution": [calibration.width, calibration.height],
            "calibration_resolution_matches_every_decoded_image": calibration_matches_images,
            "estimated_operational_offset_ns": offset_ns,
            "offset_estimator_difference_summary_ns": {
                "median": int(np.median(estimator_differences)),
                "minimum": min(estimator_differences),
                "maximum": max(estimator_differences),
            },
        }

    calibration_report = {}
    for channel, calibration in calibrations.items():
        calibration_report[channel] = {
            "intrinsic_k": calibration.intrinsic.astype(float).tolist(),
            "image_width": calibration.width,
            "image_height": calibration.height,
            "camera_model": calibration.camera_model,
            "distortion_model": calibration.distortion_model,
            "distortion_coefficients": calibration.distortion_coefficients.astype(float).tolist(),
            "coefficient_order": list(calibration.coefficient_order),
            "rectification_matrix": calibration.rectification.astype(float).tolist(),
            "projection_matrix": calibration.projection.astype(float).tolist(),
            "optical_frame": calibration.optical_frame,
            "directed_transform_path": list(calibration.directed_frame_path),
            "transform_direction": (
                f"T_base_link_{calibration.optical_frame}; maps points from the optical "
                "camera frame into base_link"
            ),
            "translation_m": calibration.sensor_to_base.translation.astype(float).tolist(),
            "rotation_wxyz": calibration.sensor_to_base.rotation_wxyz.astype(float).tolist(),
            "quaternion_norm": float(np.linalg.norm(calibration.sensor_to_base.rotation_wxyz)),
            "image_state": calibration.image_state,
            "image_state_evidence": list(calibration.image_state_evidence),
            "calibration_source": calibration.calibration_source,
        }

    blockers = []
    for channel, report in stream_reports.items():
        if (
            report["duplicate_filenames"]
            or report["duplicate_timestamps_ns"]
            or report["unparseable_supported_extension_filenames"]
            or report["unreadable_files"]
            or not report["calibration_resolution_matches_every_decoded_image"]
        ):
            blockers.append(channel)
    report = {
        "mode": "read-only camera discovery",
        "source_root": str(source_root),
        "base_pilot": str(base_pilot),
        "input_version": config["input_version"],
        "scenario": scenario,
        "resolved_source_recordings": [str(scene_path)],
        "public_scenes": [scenes[0]["name"]],
        "specification_wording_resolution": {
            "text": "Each conversion input contains one source recording and its scene-local instances.",
            "implemented_scope": "one source recording / one public scene staging unit",
            "source_instance_count": len(instances),
        },
        "lidar_anchor": {
            "source_directory": str(scene_path / "sensor_data" / "lidar"),
            "source_files": len(all_lidar_files),
            "included_annotated_safe_pose_files": len(lidar_files),
            "orphan_or_pose_excluded_files": len(all_lidar_files) - len(lidar_files),
            "input_pilot_records": len(lidar_records),
            "unparseable_filenames": lidar_unparseable,
            "first_timestamp_ns": lidar_files[0].timestamp_ns,
            "last_timestamp_ns": lidar_files[-1].timestamp_ns,
        },
        "camera_streams": stream_reports,
        "calibration": {
            "intrinsics_path": str(intrinsics_path),
            "extrinsics_path": str(extrinsics_path),
            "release_wide_calibration_files_found": global_calibration_files,
            "constancy_interpretation": "One release-level calibration pair; no per-recording alternatives exist.",
            "cameras": calibration_report,
        },
        "clock_interpretation": {
            "documented_shared_hardware_clock": False,
            "known_independent_clock_bias": False,
            "status": "producer clock-domain relationship not documented",
            "safe_policy": (
                "Use the required per-camera median offset only for temporary matching; "
                "preserve source camera timestamps publicly and do not claim hardware synchronization."
            ),
        },
        "per_measurement_pose_exclusion_count": sum(
            len(value) for value in camera_pose_exclusions.values()
        ),
        "discovery_blockers": blockers,
        "decision": "PASS" if not blockers else "FAIL/BLOCKED",
        "source_tree_written": False,
    }
    context = {
        "scene_path": scene_path,
        "public_scene": scenes[0]["name"],
        "lidar_files": lidar_files,
        "camera_files": camera_files,
        "calibrations": calibrations,
        "pose_stream": pose_stream,
        "camera_pose_exclusions": camera_pose_exclusions,
        "base_tables": {
            path.stem: _load_json(path)
            for path in sorted(version_root.glob("*.json"))
        },
    }
    return report, context
