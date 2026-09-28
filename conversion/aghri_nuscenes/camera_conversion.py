"""One-scene camera–LiDAR materialization."""

from __future__ import annotations

from bisect import bisect_left
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from . import __version__
from .annotations import sha256_file
from .camera_sync import synchronize
from .converter import _write_json, _write_jsonl, _write_public_table_json
from .poses import compare_tf_composition, load_pose_stream, rebase_pose
from .timestamps import nanoseconds_to_microseconds
from .tokens import deterministic_token


OFFICIAL_TABLE_NAMES = (
    "category", "attribute", "visibility", "sensor", "calibrated_sensor",
    "ego_pose", "log", "scene", "sample", "sample_data", "instance",
    "sample_annotation", "map", "splits",
)


def _safe_copy(source: Path, destination: Path) -> str:
    if os.environ.get("AGHRI_NUSCENES_METADATA_ONLY") == "1":
        return sha256_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    byte_count = 0
    with source.open("rb") as input_handle, destination.open("wb") as output_handle:
        for block in iter(lambda: input_handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
            output_handle.write(block)
            byte_count += len(block)
    source_size = source.stat().st_size
    if byte_count != source_size or destination.stat().st_size != source_size:
        raise ValueError(f"byte-identity copy failed: {source} -> {destination}")
    return digest.hexdigest()


def _calibration_token(namespace: str, channel: str, optical_frame: str) -> str:
    return deterministic_token(
        namespace, f"calibrated_sensor|{channel}|base_link|{optical_frame}"
    )


def _camera_model_record(namespace: str, calibration, calibration_token: str) -> dict:
    return {
        "token": deterministic_token(namespace, f"camera_model|{calibration_token}"),
        "calibrated_sensor_token": calibration_token,
        "channel": calibration.channel,
        "image_width": calibration.width,
        "image_height": calibration.height,
        "camera_model": calibration.camera_model,
        "distortion_model": calibration.distortion_model,
        "distortion_coefficients": calibration.distortion_coefficients.astype(float).tolist(),
        "coefficient_order": list(calibration.coefficient_order),
        "image_state": calibration.image_state,
        "rectification_matrix": calibration.rectification.astype(float).tolist(),
        "projection_matrix": calibration.projection.astype(float).tolist(),
        "calibration_source": calibration.calibration_source,
    }


def _link_camera_stream(records: list[dict]) -> None:
    for index, record in enumerate(records):
        record["prev"] = records[index - 1]["token"] if index else ""
        record["next"] = records[index + 1]["token"] if index + 1 < len(records) else ""


def _hash_public_tables(version_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in OFFICIAL_TABLE_NAMES + ("camera_model",):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update((version_dir / f"{name}.json").read_bytes())
    return digest.hexdigest()


def convert_camera_pilot(
    source_root: Path,
    base_pilot: Path,
    output_root: Path,
    scenario: str,
    config: dict,
    discovery_report: dict,
    context: dict,
) -> dict:
    """Create a camera–LiDAR conversion only in a new empty directory."""
    if discovery_report["decision"] != "PASS":
        raise ValueError(
            f"camera discovery has blockers: {discovery_report['discovery_blockers']}"
        )
    if output_root.exists():
        if not output_root.is_dir() or any(output_root.iterdir()):
            raise ValueError(f"output root already exists and is not empty: {output_root}")
    else:
        output_root.mkdir(parents=True)

    version_dir = output_root / config["version"]
    manifests_dir = output_root / "manifests"
    reports_dir = output_root / "reports"
    sync_dir = reports_dir / "sync"
    for directory in (version_dir, manifests_dir, sync_dir):
        directory.mkdir(parents=True, exist_ok=True)

    base_tables = context["base_tables"]
    public_scene = context["public_scene"]
    lidar_files = context["lidar_files"]
    camera_files = context["camera_files"]
    camera_pose_exclusions = context["camera_pose_exclusions"]
    calibrations = context["calibrations"]
    camera_definitions = config["camera_channels"]
    channel_order = [definition["channel"] for definition in camera_definitions]
    namespace = config["uuid_namespace"]

    sync_cfg = config["synchronization"]
    eligible_camera_files = {
        channel: [
            item for item in camera_files[channel]
            if item.name not in camera_pose_exclusions[channel]
        ]
        for channel in channel_order
    }
    sync_result = synchronize(
        lidar_files,
        eligible_camera_files,
        [float(value) for value in sync_cfg["thresholds_s"]],
        float(sync_cfg["p95_limit_s"]),
        float(sync_cfg["plateau_eps"]),
        offset_camera_files={
            channel: camera_files[channel] for channel in channel_order
        },
    )
    sync_result["recording"] = scenario
    sync_result["public_scene"] = public_scene
    _write_json(sync_dir / f"{public_scene}_sync.json", sync_result)
    sync_summary = {
        "anchor": "lidar",
        "recordings": [{
            "recording": scenario,
            "public_scene": public_scene,
            "chosen_threshold_s": sync_result["chosen_threshold_s"],
            "p95_preference_met": sync_result["p95_preference_met"],
            "estimated_camera_offsets_ns": sync_result["estimated_camera_offsets_ns"],
            "chosen_threshold_metrics": sync_result["chosen_threshold_metrics"],
        }],
    }
    _write_json(reports_dir / "synchronization_summary.json", sync_summary)
    _write_json(reports_dir / "camera_discovery.json", discovery_report)

    samples = [dict(record) for record in base_tables["sample"]]
    if len(samples) != len(lidar_files):
        raise ValueError("LiDAR source and sample counts differ")
    for sample, lidar_file in zip(samples, lidar_files):
        expected_us, _ = nanoseconds_to_microseconds(lidar_file.timestamp_ns)
        if sample["timestamp"] != expected_us:
            raise ValueError("base sample order does not match source LiDAR timestamps")
    lidar_ns = [item.timestamp_ns for item in lidar_files]
    chosen_threshold_ns = int(round(sync_result["chosen_threshold_s"] * 1_000_000_000))

    selected_keyframes: dict[str, dict[str, int]] = {channel: {} for channel in channel_order}
    completeness_rows: list[dict] = []
    for row in sync_result["samples"]:
        available = {}
        missing = []
        keyframe_tokens = {}
        for channel in channel_order:
            filename = row["cameras"][channel]["selected_filename"]
            available[channel] = filename is not None
            if filename is None:
                missing.append(channel)
            else:
                if filename in selected_keyframes[channel]:
                    raise ValueError(f"camera keyframe reused for {channel}: {filename}")
                selected_keyframes[channel][filename] = row["lidar_index"]
        completeness_rows.append({
            "sample_index": row["lidar_index"],
            "sample_token": samples[row["lidar_index"]]["token"],
            "lidar_timestamp_ns": row["lidar_timestamp_ns"],
            "camera_keyframes_present": available,
            "missing_channels": missing,
            "complete_four_camera_sample": not missing,
            "camera_sample_data_tokens": keyframe_tokens,
        })

    pose_cfg = config["pose"]
    scene_path = context["scene_path"]
    global_stream = context["pose_stream"]
    map_odom_stream = load_pose_stream(
        scene_path / pose_cfg["map_odom_tf"], "tr", "map_to_odom"
    )
    odom_base_stream = load_pose_stream(
        scene_path / pose_cfg["odom_base_tf"], "tr", "odom_to_base_link"
    )
    first_global_pose = global_stream.interpolate(
        lidar_files[0].timestamp_ns, float(pose_cfg["maximum_bracket_ms"])
    ).pose

    camera_sample_data: list[dict] = []
    camera_ego_poses: list[dict] = []
    disposition_records: list[dict] = []
    pose_manifest: list[dict] = []
    included_by_channel: dict[str, list[dict]] = {channel: [] for channel in channel_order}
    output_by_channel_role: dict[str, dict[str, int]] = {
        channel: {"keyframe": 0, "sweep": 0, "excluded": 0}
        for channel in channel_order
    }
    maximum_pose_bracket_ms = 0.0
    maximum_tf_translation_error_m = 0.0
    maximum_tf_rotation_error_rad = 0.0

    sensor_records = [
        dict(record) for record in base_tables["sensor"]
        if record["channel"] != "cam_zed_depth"
    ]
    active_channels = {"lidar", *channel_order}
    if {record["channel"] for record in sensor_records} != active_channels:
        raise ValueError("active sensor table does not resolve to LiDAR plus four RGB cameras")
    sensor_tokens = {record["channel"]: record["token"] for record in sensor_records}
    calibration_records = [dict(record) for record in base_tables["calibrated_sensor"]]
    camera_model_records = []
    calibration_tokens: dict[str, str] = {}
    for channel in channel_order:
        calibration = calibrations[channel]
        calibration_token = _calibration_token(namespace, channel, calibration.optical_frame)
        calibration_tokens[channel] = calibration_token
        calibration_records.append({
            "token": calibration_token,
            "sensor_token": sensor_tokens[channel],
            "translation": calibration.sensor_to_base.translation.astype(float).tolist(),
            "rotation": calibration.sensor_to_base.rotation_wxyz.astype(float).tolist(),
            "camera_intrinsic": calibration.intrinsic.astype(float).tolist(),
        })
        camera_model_records.append(
            _camera_model_record(namespace, calibration, calibration_token)
        )

    for channel in channel_order:
        calibration = calibrations[channel]
        for item in camera_files[channel]:
            if item.name in camera_pose_exclusions[channel]:
                role = "excluded"
                sample_index = None
                exclusion_reason = camera_pose_exclusions[channel][item.name]
            elif item.name in selected_keyframes[channel]:
                role = "keyframe"
                sample_index = selected_keyframes[channel][item.name]
                exclusion_reason = None
            else:
                following = bisect_left(lidar_ns, item.timestamp_ns)
                if following == len(lidar_ns):
                    role = "excluded"
                    sample_index = None
                    exclusion_reason = "no_following_lidar_sample_within_scene"
                elif lidar_ns[following] - item.timestamp_ns > chosen_threshold_ns:
                    role = "excluded"
                    sample_index = None
                    exclusion_reason = "following_sample_exceeds_large_gap_threshold"
                else:
                    role = "sweep"
                    sample_index = following
                    exclusion_reason = None
            output_by_channel_role[channel][role] += 1
            if role == "excluded":
                source_hash = sha256_file(item.path)
                disposition_records.append({
                    "channel": channel,
                    "source_filename": item.name,
                    "source_timestamp_ns": item.timestamp_ns,
                    "source_sha256": source_hash,
                    "disposition": role,
                    "reason": exclusion_reason,
                    "sample_index": None,
                    "sample_token": None,
                    "output_relative_path": None,
                    "output_sha256": None,
                })
                continue

            global_interp = global_stream.interpolate(
                item.timestamp_ns, float(pose_cfg["maximum_bracket_ms"])
            )
            map_odom_interp = map_odom_stream.interpolate(
                item.timestamp_ns, float(pose_cfg["maximum_bracket_ms"])
            )
            odom_base_interp = odom_base_stream.interpolate(
                item.timestamp_ns, float(pose_cfg["maximum_bracket_ms"])
            )
            translation_error, rotation_error = compare_tf_composition(
                global_interp.pose, map_odom_interp.pose, odom_base_interp.pose
            )
            if translation_error > float(pose_cfg["tf_composition_translation_tolerance_m"]):
                raise ValueError(f"camera TF translation disagreement: {channel}/{item.name}")
            if rotation_error > float(pose_cfg["tf_composition_rotation_tolerance_rad"]):
                raise ValueError(f"camera TF rotation disagreement: {channel}/{item.name}")
            rebased = rebase_pose(global_interp.pose, first_global_pose)
            timestamp_us, discarded_ns = nanoseconds_to_microseconds(item.timestamp_ns)
            sample = samples[sample_index]
            canonical_suffix = f"{timestamp_us}|{item.name}"
            data_token = deterministic_token(
                namespace,
                f"sample_data|{public_scene}|{channel}|{canonical_suffix}",
            )
            pose_token = deterministic_token(
                namespace,
                f"ego_pose|{public_scene}|{channel}|{canonical_suffix}",
            )
            output_relative = (
                f"{'samples' if role == 'keyframe' else 'sweeps'}/{channel}/"
                f"{public_scene}__{item.name}"
            )
            output_hash = _safe_copy(item.path, output_root / output_relative)
            source_hash = output_hash
            record = {
                "token": data_token,
                "sample_token": sample["token"],
                "ego_pose_token": pose_token,
                "calibrated_sensor_token": calibration_tokens[channel],
                "timestamp": timestamp_us,
                "fileformat": item.extension.lstrip("."),
                "is_key_frame": role == "keyframe",
                "height": calibration.height,
                "width": calibration.width,
                "filename": output_relative,
                "prev": "",
                "next": "",
            }
            camera_sample_data.append(record)
            included_by_channel[channel].append(record)
            camera_ego_poses.append({
                "token": pose_token,
                "timestamp": timestamp_us,
                "rotation": rebased.rotation_wxyz.astype(float).tolist(),
                "translation": rebased.translation.astype(float).tolist(),
            })
            maximum_pose_bracket_ms = max(maximum_pose_bracket_ms, global_interp.bracket_ms)
            maximum_tf_translation_error_m = max(
                maximum_tf_translation_error_m, translation_error
            )
            maximum_tf_rotation_error_rad = max(
                maximum_tf_rotation_error_rad, rotation_error
            )
            pose_manifest.append({
                "channel": channel,
                "source_filename": item.name,
                "source_timestamp_ns": item.timestamp_ns,
                "output_timestamp_us": timestamp_us,
                "discarded_submicrosecond_ns": discarded_ns,
                "ego_pose_token": pose_token,
                "interpolation": {
                    "left_index": global_interp.left_index,
                    "right_index": global_interp.right_index,
                    "left_time_s": global_interp.left_time_s,
                    "right_time_s": global_interp.right_time_s,
                    "amount": global_interp.amount,
                    "bracket_ms": global_interp.bracket_ms,
                },
                "tf_composition_translation_error_m": translation_error,
                "tf_composition_rotation_error_rad": rotation_error,
                "rebased_translation": rebased.translation.astype(float).tolist(),
                "rebased_rotation_wxyz": rebased.rotation_wxyz.astype(float).tolist(),
            })
            disposition = {
                "channel": channel,
                "source_filename": item.name,
                "source_timestamp_ns": item.timestamp_ns,
                "source_sha256": source_hash,
                "disposition": role,
                "reason": None,
                "sample_index": sample_index,
                "sample_token": sample["token"],
                "output_relative_path": output_relative,
                "output_sha256": output_hash,
                "sample_data_token": data_token,
                "ego_pose_token": pose_token,
            }
            disposition_records.append(disposition)
            if role == "keyframe":
                completeness_rows[sample_index]["camera_sample_data_tokens"][channel] = data_token

    for channel in channel_order:
        _link_camera_stream(included_by_channel[channel])

    for record in completeness_rows:
        if record["complete_four_camera_sample"]:
            if set(record["camera_sample_data_tokens"]) != set(channel_order):
                raise ValueError("complete sample lacks four camera sample_data tokens")
        elif set(record["camera_sample_data_tokens"]) != (
            set(channel_order) - set(record["missing_channels"])
        ):
            raise ValueError("incomplete sample keyframe-token manifest mismatch")

    output_tables = {
        name: [dict(record) for record in value] if isinstance(value, list) else dict(value)
        for name, value in base_tables.items()
        if name in OFFICIAL_TABLE_NAMES
    }
    output_tables["sensor"] = sensor_records
    output_tables["calibrated_sensor"] = calibration_records
    output_tables["ego_pose"] = [
        dict(record) for record in base_tables["ego_pose"]
    ] + camera_ego_poses
    output_tables["sample_data"] = [
        dict(record) for record in base_tables["sample_data"]
    ] + camera_sample_data
    for name in OFFICIAL_TABLE_NAMES:
        _write_public_table_json(version_dir / f"{name}.json", name, output_tables[name])
    _write_public_table_json(
        version_dir / "camera_model.json", "camera_model", camera_model_records
    )

    base_lidar_dir = base_pilot / "samples" / "lidar"
    for path in sorted(base_lidar_dir.glob("*.pcd.bin")):
        _safe_copy(path, output_root / "samples" / "lidar" / path.name)

    base_manifest_dir = base_pilot / "manifests"
    for name in ("conversion_manifest.jsonl", "exclusions.jsonl", "source_to_scene.json"):
        _safe_copy(base_manifest_dir / name, manifests_dir / name)
    _safe_copy(
        base_manifest_dir / "conversion_config.json",
        manifests_dir / "lidar_conversion_config.json",
    )
    _write_jsonl(manifests_dir / "camera_file_disposition.jsonl", disposition_records)
    _write_jsonl(manifests_dir / "camera_ego_pose_manifest.jsonl", pose_manifest)
    completeness_summary = {
        "public_scene": public_scene,
        "total_lidar_samples": len(samples),
        "complete_four_camera_samples": sum(
            row["complete_four_camera_sample"] for row in completeness_rows
        ),
        "incomplete_samples": sum(
            not row["complete_four_camera_sample"] for row in completeness_rows
        ),
        "per_channel_keyframes": {
            channel: len(selected_keyframes[channel]) for channel in channel_order
        },
        "samples": completeness_rows,
    }
    _write_json(manifests_dir / "camera_completeness.json", completeness_summary)

    conversion_config = {
        "converter_version": __version__,
        "operation": "camera–LiDAR conversion",
        "scenario": scenario,
        "public_scene": public_scene,
        "base_pilot_version": config["input_version"],
        "output_version": config["version"],
        "base_pilot_sha256_scope": "protected before/after inventories outside dataset",
        "source_tree_written": False,
        "source_lidar_conversion_written": False,
        "zed_depth_excluded": True,
        "images_modified": False,
        "public_camera_timestamp_policy": sync_cfg["public_timestamp_policy"],
        "offset_usage": sync_cfg["offset_usage"],
        "active_sensor_channels": ["lidar", *channel_order],
        "removed_inactive_sensor_placeholder": "cam_zed_depth",
        "camera_calibrations": {
            channel: {
                "calibrated_sensor_token": calibration_tokens[channel],
                "camera_model_token": camera_model_records[index]["token"],
                "optical_frame": calibrations[channel].optical_frame,
                "directed_transform_path": list(calibrations[channel].directed_frame_path),
                "translation_m": calibrations[channel].sensor_to_base.translation.astype(float).tolist(),
                "rotation_wxyz": calibrations[channel].sensor_to_base.rotation_wxyz.astype(float).tolist(),
                "coefficient_order": list(calibrations[channel].coefficient_order),
            }
            for index, channel in enumerate(channel_order)
        },
        "counts": {
            "lidar_samples": len(samples),
            "camera_source_files": sum(len(camera_files[channel]) for channel in channel_order),
            "camera_sample_data": len(camera_sample_data),
            "camera_keyframes": sum(
                counts["keyframe"] for counts in output_by_channel_role.values()
            ),
            "camera_sweeps": sum(
                counts["sweep"] for counts in output_by_channel_role.values()
            ),
            "camera_exclusions": sum(
                counts["excluded"] for counts in output_by_channel_role.values()
            ),
            "camera_ego_poses": len(camera_ego_poses),
            "camera_models": len(camera_model_records),
        },
        "per_channel_disposition": output_by_channel_role,
        "pose_evidence": {
            "maximum_bracket_ms": maximum_pose_bracket_ms,
            "maximum_tf_translation_error_m": maximum_tf_translation_error_m,
            "maximum_tf_rotation_error_rad": maximum_tf_rotation_error_rad,
        },
    }
    _write_json(manifests_dir / "conversion_config.json", conversion_config)
    conversion_config["aggregate_public_table_file_hash"] = _hash_public_tables(version_dir)
    _write_json(manifests_dir / "conversion_config.json", conversion_config)
    return {
        "sync": sync_result,
        "tables": output_tables,
        "camera_model": camera_model_records,
        "dispositions": disposition_records,
        "completeness": completeness_summary,
        "conversion_config": conversion_config,
    }
