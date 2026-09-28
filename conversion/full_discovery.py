"""Read-only discovery gate for all 65 authoritative AGHRI recordings."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import traceback

import numpy as np
import yaml

from aghri_nuscenes.annotations import (
    load_annotation_records,
    resolve_annotation_source,
    sha256_file,
    validate_box,
)
from aghri_nuscenes.camera_calibration import load_camera_calibrations
from aghri_nuscenes.camera_discovery import _inspect_images
from aghri_nuscenes.camera_sync import estimate_constant_offset_ns, scan_timed_files
from aghri_nuscenes.discovery import resolve_all_scenes, resolve_annotation_pcd_references
from aghri_nuscenes.pcd import parse_pcd_header
from aghri_nuscenes.poses import (
    compare_tf_composition,
    load_pose_stream,
    resolve_lidar_calibration,
)
from aghri_nuscenes.timestamps import parse_pcd_filename


CAMERA_CHANNELS = (
    "cam_zed_rgb",
    "cam_fish_front",
    "cam_fish_left",
    "cam_fish_right",
)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=False, allow_nan=False) + "\n")


def _max_gap_ns(values: list[int]) -> int | None:
    if len(values) < 2:
        return None
    return max(right - left for left, right in zip(values, values[1:]))


def _camera_definition(config: dict, channel: str) -> dict:
    matches = [item for item in config["camera_channels"] if item["channel"] == channel]
    if len(matches) != 1:
        raise ValueError("camera definition does not resolve once: {}".format(channel))
    return matches[0]


def _pose_audit(streams: tuple, timestamp_ns: int, pose_cfg: dict) -> tuple[dict | None, dict | None]:
    try:
        global_interp = streams[0].interpolate(timestamp_ns, float(pose_cfg["maximum_bracket_ms"]))
        map_odom = streams[1].interpolate(timestamp_ns, float(pose_cfg["maximum_bracket_ms"]))
        odom_base = streams[2].interpolate(timestamp_ns, float(pose_cfg["maximum_bracket_ms"]))
        translation_error, rotation_error = compare_tf_composition(
            global_interp.pose, map_odom.pose, odom_base.pose
        )
        if translation_error > float(pose_cfg["tf_composition_translation_tolerance_m"]):
            raise ValueError("TF translation composition tolerance exceeded")
        if rotation_error > float(pose_cfg["tf_composition_rotation_tolerance_rad"]):
            raise ValueError("TF rotation composition tolerance exceeded")
        return {
            "bracket_ms": float(global_interp.bracket_ms),
            "tf_translation_error_m": float(translation_error),
            "tf_rotation_error_rad": float(rotation_error),
        }, None
    except Exception as error:
        return None, {
            "timestamp_ns": int(timestamp_ns),
            "error_type": type(error).__name__,
            "error": str(error),
        }


def discover_scene(source_root: Path, resolution, lidar_config: dict, camera_config: dict,
                   overrides: dict, calibrations: dict) -> dict:
    scene_path = source_root / resolution.source_relative_path
    override = overrides.get("scenes", {}).get(resolution.source_name)
    annotation_path, reason, candidates = resolve_annotation_source(
        scene_path, resolution.source_name, override
    )
    source_records = load_annotation_records(annotation_path)
    lidar_dir = scene_path / "sensor_data/lidar"
    available_pcd = {path.name: path for path in sorted(lidar_dir.glob("*.pcd"))}
    repair_policy = (override or {}).get("filename_reference_repair")
    records, reference_corrections = resolve_annotation_pcd_references(
        source_records, available_pcd, repair_policy
    )
    referenced = []
    timestamps = []
    valid_boxes = 0
    malformed_boxes = []
    nonzero_rotation_boxes = 0
    track_names = set()
    for record_index, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(record.get("File"), str):
            raise ValueError("invalid annotation record {}".format(record_index))
        filename = record["File"]
        _, _, timestamp_ns = parse_pcd_filename(filename)
        referenced.append(filename)
        timestamps.append(timestamp_ns)
        if filename not in available_pcd:
            raise ValueError("annotation references missing PCD: {}".format(filename))
        parse_pcd_header(available_pcd[filename])
        seen_in_frame = set()
        for label_index, label in enumerate(record.get("Labels", [])):
            box, exclusion = validate_box(record_index, label_index, label)
            if exclusion:
                malformed_boxes.append(exclusion)
                continue
            if box.source_identity in seen_in_frame:
                raise ValueError("duplicate track identity in frame: {}".format(box.source_identity))
            seen_in_frame.add(box.source_identity)
            track_names.add(box.source_identity)
            valid_boxes += 1
            nonzero_rotation_boxes += int(any(value != 0.0 for value in box.raw_rotation))
    if timestamps != sorted(timestamps) or len(timestamps) != len(set(timestamps)):
        raise ValueError("annotated LiDAR timestamps are not strictly increasing and unique")
    if len(referenced) != len(set(referenced)):
        raise ValueError("annotation references a PCD more than once")

    pose_cfg = lidar_config["pose"]
    streams = (
        load_pose_stream(scene_path / pose_cfg["source"], "p", "odom_global"),
        load_pose_stream(scene_path / pose_cfg["map_odom_tf"], "tr", "map_to_odom"),
        load_pose_stream(scene_path / pose_cfg["odom_base_tf"], "tr", "odom_to_base_link"),
    )
    lidar_pose_valid = []
    lidar_pose_invalid = []
    for index, timestamp_ns in enumerate(timestamps):
        evidence, error = _pose_audit(streams, timestamp_ns, pose_cfg)
        if error:
            lidar_pose_invalid.append({"record_index": index, "filename": referenced[index], **error})
        else:
            lidar_pose_valid.append(evidence)

    lidar_ns = np.asarray(timestamps, dtype=np.int64)
    camera_reports = {}
    for channel in CAMERA_CHANNELS:
        definition = _camera_definition(camera_config, channel)
        directory = scene_path / definition["source_path"]
        camera_files, unparseable = scan_timed_files(directory, definition["extensions"])
        if not directory.is_dir() or not camera_files:
            raise ValueError("missing/empty camera stream: {}".format(channel))
        timestamp_values = [item.timestamp_ns for item in camera_files]
        timestamp_counts = Counter(timestamp_values)
        duplicate_timestamps = sorted(value for value, count in timestamp_counts.items() if count > 1)
        filename_counts = Counter(item.name for item in camera_files)
        duplicate_filenames = sorted(value for value, count in filename_counts.items() if count > 1)
        supported = {item.path for item in camera_files}
        other_files = sorted(path.name for path in directory.iterdir() if path.is_file() and path not in supported)
        image_report = _inspect_images(camera_files)
        pose_valid = []
        pose_invalid = []
        for item in camera_files:
            evidence, error = _pose_audit(streams, item.timestamp_ns, camera_config["pose"])
            if error:
                pose_invalid.append({"filename": item.name, **error})
            else:
                pose_valid.append(evidence)
        offset_ns, nearest_differences = estimate_constant_offset_ns(
            lidar_ns, np.asarray(timestamp_values, dtype=np.int64)
        )
        calibration = calibrations[channel]
        layouts = image_report["observed_image_layouts"]
        layout_ok = (
            len(layouts) == 1
            and layouts[0]["width"] == calibration.width
            and layouts[0]["height"] == calibration.height
        )
        camera_reports[channel] = {
            "source_directory": directory.relative_to(source_root).as_posix(),
            "file_count": len(camera_files),
            "first_timestamp_ns": int(timestamp_values[0]),
            "last_timestamp_ns": int(timestamp_values[-1]),
            "maximum_consecutive_gap_ns": _max_gap_ns(timestamp_values),
            "extensions": dict(sorted(Counter(item.extension for item in camera_files).items())),
            "unparseable_supported_filenames": unparseable,
            "duplicate_timestamps_ns": duplicate_timestamps,
            "duplicate_filenames": duplicate_filenames,
            "other_files": other_files,
            **image_report,
            "calibration_resolution": [calibration.width, calibration.height],
            "calibration_resolution_matches_all_images": layout_ok,
            "pose_valid_files": len(pose_valid),
            "pose_invalid_files": pose_invalid,
            "maximum_valid_pose_bracket_ms": max(
                (item["bracket_ms"] for item in pose_valid), default=None
            ),
            "estimated_operational_offset_ns": int(offset_ns),
            "nearest_difference_ns": {
                "minimum": int(min(nearest_differences)),
                "median": int(np.median(nearest_differences)),
                "maximum": int(max(nearest_differences)),
            },
        }

    hard_blockers = []
    if not lidar_pose_valid:
        hard_blockers.append("no_safe_annotated_lidar_pose")
    for channel, report in camera_reports.items():
        if (
            report["unparseable_supported_filenames"]
            or report["duplicate_timestamps_ns"]
            or report["duplicate_filenames"]
            or report["unreadable_files"]
            or not report["calibration_resolution_matches_all_images"]
        ):
            hard_blockers.append("camera_integrity:{}".format(channel))
    return {
        "public_scene": resolution.public_scene_name,
        "source_name": resolution.source_name,
        "source_relative_path": resolution.source_relative_path,
        "split": resolution.split,
        "source_file_count": sum(1 for path in scene_path.rglob("*") if path.is_file()),
        "selected_annotation": annotation_path.relative_to(source_root).as_posix(),
        "selected_annotation_sha256": sha256_file(annotation_path),
        "annotation_selection_reason": reason,
        "annotation_candidates": candidates,
        "annotation_records": len(records),
        "annotation_filename_reference_repairs": reference_corrections,
        "valid_source_boxes": valid_boxes,
        "malformed_source_boxes": malformed_boxes,
        "source_boxes_with_nonzero_rotation_values": nonzero_rotation_boxes,
        "scene_local_source_track_count": len(track_names),
        "lidar": {
            "directory": lidar_dir.relative_to(source_root).as_posix(),
            "discovered_pcd_files": len(available_pcd),
            "annotated_pcd_files": len(referenced),
            "orphan_nonannotated_pcd_files": len(set(available_pcd) - set(referenced)),
            "orphan_nonannotated_pcd_names": sorted(set(available_pcd) - set(referenced)),
            "first_annotated_timestamp_ns": int(timestamps[0]),
            "last_annotated_timestamp_ns": int(timestamps[-1]),
            "maximum_consecutive_annotated_gap_ns": _max_gap_ns(timestamps),
            "safe_pose_measurements": len(lidar_pose_valid),
            "unsafe_pose_measurements": lidar_pose_invalid,
            "maximum_valid_pose_bracket_ms": max(
                (item["bracket_ms"] for item in lidar_pose_valid), default=None
            ),
        },
        "pose_sources": {
            "global": pose_cfg["source"],
            "map_to_odom": pose_cfg["map_odom_tf"],
            "odom_to_base_link": pose_cfg["odom_base_tf"],
            "global_time_range_s": list(streams[0].time_range_s),
        },
        "cameras": camera_reports,
        "hard_blockers": hard_blockers,
        "decision": "PASS" if not hard_blockers else "BLOCKED",
    }


def discover(source_root: Path, split_root: Path, lidar_config_path: Path,
             camera_config_path: Path, overrides_path: Path) -> dict:
    lidar_config = yaml.safe_load(lidar_config_path.read_text())
    camera_config = yaml.safe_load(camera_config_path.read_text())
    overrides = yaml.safe_load(overrides_path.read_text()) or {}
    resolutions, resolution_report = resolve_all_scenes(source_root, split_root)
    calibrations = load_camera_calibrations(
        source_root / camera_config["calibration"]["intrinsics"],
        source_root / camera_config["calibration"]["extrinsics"],
        camera_config["camera_channels"],
    )
    _, lidar_calibration = resolve_lidar_calibration(
        source_root / lidar_config["calibration"]["source"], lidar_config["calibration"]
    )
    scenes = []
    errors = []
    for index, resolution in enumerate(resolutions, start=1):
        try:
            scene = discover_scene(
                source_root, resolution, lidar_config, camera_config, overrides, calibrations
            )
            scenes.append(scene)
            print("DISCOVERY {}/65 {} {}".format(index, resolution.public_scene_name, scene["decision"]), flush=True)
        except Exception as error:
            errors.append({
                "public_scene": resolution.public_scene_name,
                "source_name": resolution.source_name,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            })
            print("DISCOVERY {}/65 {} ERROR {}".format(index, resolution.public_scene_name, error), flush=True)
    split_counts = Counter(scene["split"] for scene in scenes)
    camera_totals = {
        channel: sum(scene["cameras"][channel]["file_count"] for scene in scenes)
        for channel in CAMERA_CHANNELS
    }
    blockers = errors + [
        {"public_scene": scene["public_scene"], "hard_blockers": scene["hard_blockers"]}
        for scene in scenes if scene["hard_blockers"]
    ]
    return {
        "operation": "AGHRI full read-only conversion discovery",
        "decision": "PASS" if not blockers and len(scenes) == 65 else "BLOCKED",
        "source_root": str(source_root),
        "split_root": str(split_root),
        "resolution": resolution_report,
        "resolved_scene_count": len(scenes),
        "resolved_split_counts": dict(split_counts),
        "camera_file_totals": camera_totals,
        "annotated_lidar_total": sum(scene["lidar"]["annotated_pcd_files"] for scene in scenes),
        "discovered_lidar_total": sum(scene["lidar"]["discovered_pcd_files"] for scene in scenes),
        "valid_box_total": sum(scene["valid_source_boxes"] for scene in scenes),
        "malformed_box_total": sum(len(scene["malformed_source_boxes"]) for scene in scenes),
        "nonzero_source_rotation_value_box_total": sum(
            scene["source_boxes_with_nonzero_rotation_values"] for scene in scenes
        ),
        "unsafe_lidar_pose_total": sum(
            len(scene["lidar"]["unsafe_pose_measurements"]) for scene in scenes
        ),
        "unsafe_camera_pose_total": sum(
            len(scene["cameras"][channel]["pose_invalid_files"])
            for scene in scenes for channel in CAMERA_CHANNELS
        ),
        "annotation_candidate_multiplicity": dict(Counter(len(scene["annotation_candidates"]) for scene in scenes)),
        "calibration": {
            "lidar": lidar_calibration,
            "camera_channels": {
                channel: {
                    "image_size": [calibrations[channel].width, calibrations[channel].height],
                    "camera_model": calibrations[channel].camera_model,
                    "distortion_model": calibrations[channel].distortion_model,
                    "coefficient_order": list(calibrations[channel].coefficient_order),
                    "directed_transform_path": list(calibrations[channel].directed_frame_path),
                }
                for channel in CAMERA_CHANNELS
            },
        },
        "blockers": blockers,
        "scenes": scenes,
        "source_tree_written": False,
    }


def write_plan(path: Path, report: dict) -> None:
    lines = [
        "# AGHRI-to-nuScenes conversion plan",
        "",
        "Discovery status: **{}**".format(report["decision"]),
        "",
        "The full converter will process the authoritative recordings in train, validation, then test order. "
        "Each recording becomes one scene. LiDAR annotations are converted first, synchronization is selected "
        "independently per recording, and each camera image is then classified once as keyframe, sweep, or exclusion.",
        "",
        "- Resolved scenes: {} ({} train / {} validation / {} test)".format(
            report["resolved_scene_count"],
            report["resolved_split_counts"].get("train", 0),
            report["resolved_split_counts"].get("val", 0),
            report["resolved_split_counts"].get("test", 0),
        ),
        "- Annotated LiDAR frames discovered: {}".format(report["annotated_lidar_total"]),
        "- Valid source boxes discovered: {}".format(report["valid_box_total"]),
        "- Camera files discovered: {}".format(sum(report["camera_file_totals"].values())),
        "- ZED depth: excluded",
        "- Images: byte-identical raw copies",
        "- Maps/radar/lidarseg/panoptic: not created",
        "- Source box rotation policy: identity; original values retained in audit evidence",
        "- Output version: `v1.0-aghri`",
        "",
        "Materialization is permitted only while discovery remains PASS. PKL generation should begin only after conversion completes successfully.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--lidar-config", type=Path, required=True)
    parser.add_argument("--camera-config", type=Path, required=True)
    parser.add_argument("--annotation-overrides", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve(strict=True)
    split = args.split_root.resolve(strict=True)
    output = args.output.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("unsafe discovery output overlap")
    report = discover(source, split, args.lidar_config, args.camera_config, args.annotation_overrides)
    _write_json(output, report)
    write_plan(args.plan, report)
    print(json.dumps({key: report[key] for key in (
        "decision", "resolved_scene_count", "resolved_split_counts", "annotated_lidar_total",
        "camera_file_totals", "valid_box_total", "unsafe_lidar_pose_total",
        "unsafe_camera_pose_total", "blockers")}, indent=2))
    if report["decision"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
