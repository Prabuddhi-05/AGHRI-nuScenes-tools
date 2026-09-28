"""One-scene Stage-1 conversion orchestration."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .annotations import count_points_in_box, sha256_file, validate_box
from .discovery import describe_input_file
from .pcd import decode_aghri_pcd, encode_nuscenes_lidar
from .poses import (
    compare_tf_composition,
    load_pose_stream,
    rebase_pose,
    resolve_lidar_calibration,
)
from .public_json import order_public_table
from .render import render_visual_checks
from .tables import build_tables
from .timestamps import nanoseconds_to_microseconds, parse_pcd_filename


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _write_public_table_json(path: Path, table_name: str, value: Any) -> None:
    """Write one public table with explicit nuScenes-style field ordering."""
    ordered = order_public_table(table_name, value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(ordered, handle, indent=2, sort_keys=False, allow_nan=False)
        handle.write("\n")


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")))
            handle.write("\n")


def _input_descriptor(path: Path, root: Path) -> dict:
    value = describe_input_file(path, root)
    value["absolute_path_disclosed"] = False
    return value


def convert_pilot(
    source_root: Path,
    split_root: Path,
    output_root: Path,
    config: dict,
    plan: dict,
    context: dict,
    resolutions: list,
    discovery_report: dict,
) -> dict:
    if output_root.exists():
        if any(output_root.iterdir()) if output_root.is_dir() else True:
            raise ValueError(f"output root already exists and is not empty: {output_root}")
    else:
        output_root.mkdir(parents=True)
    version_dir = output_root / config["version"]
    payload_dir = output_root / "samples" / "lidar"
    manifest_dir = output_root / "manifests"
    report_dir = output_root / "reports"
    for directory in (version_dir, payload_dir, manifest_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    _write_json(report_dir / "pilot_plan.json", plan)
    _write_json(report_dir / "discovery_report.json", discovery_report)
    scene_path: Path = context["scene_path"]
    records: list[dict] = context["records"]
    annotation_path: Path = context["annotation_path"]
    pose_cfg = config["pose"]
    global_stream = load_pose_stream(scene_path / pose_cfg["source"], "p", "odom_global")
    map_odom_stream = load_pose_stream(scene_path / pose_cfg["map_odom_tf"], "tr", "map_to_odom")
    odom_base_stream = load_pose_stream(scene_path / pose_cfg["odom_base_tf"], "tr", "odom_to_base_link")
    extrinsics_path = source_root / config["calibration"]["source"]
    lidar_calibration, calibration_evidence = resolve_lidar_calibration(
        extrinsics_path, config["calibration"]
    )

    valid_identities: OrderedDict[str, None] = OrderedDict()
    exclusions: list[dict] = []
    validated_boxes_by_record: list[list] = []
    for record_index, record in enumerate(records):
        boxes = []
        for label_index, label in enumerate(record.get("Labels", [])):
            box, exclusion = validate_box(record_index, label_index, label)
            if exclusion:
                exclusions.append({
                    "scope": "box", "scenario": context["resolution"].public_scene_name,
                    **exclusion,
                })
                continue
            boxes.append(box)
            valid_identities.setdefault(box.source_identity, None)
        validated_boxes_by_record.append(boxes)
    identity_mapping = {
        source_identity: f"opaque-track-{index:04d}"
        for index, source_identity in enumerate(valid_identities, start=1)
    }

    processed: list[dict[str, Any]] = []
    used_sources: list[dict] = []
    static_source_paths = [
        annotation_path,
        scene_path / pose_cfg["source"],
        scene_path / pose_cfg["map_odom_tf"],
        scene_path / pose_cfg["odom_base_tf"],
        extrinsics_path,
    ]
    for path in static_source_paths:
        used_sources.append(_input_descriptor(path, source_root))
    split_inputs = [
        _input_descriptor(split_root / f"{split}.txt", split_root)
        for split in ("train", "val", "test")
    ]

    first_global_pose = None
    max_bracket_ms = float(pose_cfg["maximum_bracket_ms"])
    translation_tolerance = float(pose_cfg["tf_composition_translation_tolerance_m"])
    rotation_tolerance = float(pose_cfg["tf_composition_rotation_tolerance_rad"])
    for record_index, (record, timestamp_ns, boxes) in enumerate(
        zip(records, context["timestamp_ns"], validated_boxes_by_record)
    ):
        filename = record["File"]
        source_pcd = context["available_pcd"][filename]
        try:
            global_interp = global_stream.interpolate(timestamp_ns, max_bracket_ms)
            map_odom_interp = map_odom_stream.interpolate(timestamp_ns, max_bracket_ms)
            odom_base_interp = odom_base_stream.interpolate(timestamp_ns, max_bracket_ms)
            translation_error, rotation_error = compare_tf_composition(
                global_interp.pose, map_odom_interp.pose, odom_base_interp.pose
            )
            if translation_error > translation_tolerance or rotation_error > rotation_tolerance:
                raise ValueError(
                    f"TF composition disagreement: {translation_error} m, "
                    f"{rotation_error} rad"
                )
        except ValueError as error:
            exclusions.append({
                "scope": "lidar_measurement",
                "scenario": context["resolution"].public_scene_name,
                "record_index": record_index,
                "source_annotation_file_value": record.get(
                    "_aghri_original_file_reference", filename
                ),
                "resolved_pcd_filename": filename,
                "timestamp_ns": int(timestamp_ns),
                "reason": "unsafe_pose_interpolation_or_tf_crosscheck",
                "error": str(error),
            })
            continue
        points, header, point_stats = decode_aghri_pcd(source_pcd)
        if first_global_pose is None:
            first_global_pose = global_interp.pose
        rebased = rebase_pose(global_interp.pose, first_global_pose)
        timestamp_us, remainder_ns = nanoseconds_to_microseconds(timestamp_ns)
        seconds, nanoseconds, _ = parse_pcd_filename(filename)
        output_relative = (
            f"samples/lidar/{context['resolution'].public_scene_name}__"
            f"{seconds}_{nanoseconds:09d}.pcd.bin"
        )
        payload = encode_nuscenes_lidar(points)
        box_items = []
        for box in boxes:
            box_items.append({
                "source_box": box,
                "opaque_track": identity_mapping[box.source_identity],
                "num_lidar_pts": count_points_in_box(
                    points, box, float(config["box_policy"]["point_boundary_tolerance_m"])
                ),
            })
        source_descriptor = _input_descriptor(source_pcd, source_root)
        used_sources.append(source_descriptor)
        processed.append({
            "record_index": record_index,
            "serialized_timestamp": record.get("Timestamp"),
            "source_annotation_file_value": record.get(
                "_aghri_original_file_reference", record["File"]
            ),
            "annotation_reference_repair": record.get("_aghri_reference_repair"),
            "filename_timestamp_delta_ns_approx": (
                float(record.get("Timestamp")) * 1_000_000_000 - timestamp_ns
                if isinstance(record.get("Timestamp"), (int, float)) else None
            ),
            "timestamp_ns": timestamp_ns,
            "timestamp_us": timestamp_us,
            "discarded_submicrosecond_ns": remainder_ns,
            "source_pcd": source_pcd,
            "source_descriptor": source_descriptor,
            "output_relative_path": output_relative,
            "output_payload": payload,
            "points_xyz": points,
            "pcd_header": header.serializable(),
            "point_stats": point_stats,
            "global_pose": global_interp.pose,
            "rebased_pose": rebased,
            "pose_interpolation": global_interp,
            "map_odom_interpolation": map_odom_interp,
            "odom_base_interpolation": odom_base_interp,
            "tf_composition_translation_error_m": translation_error,
            "tf_composition_rotation_error_rad": rotation_error,
            "boxes": box_items,
        })

    if not processed:
        raise ValueError("recording has no safe annotated LiDAR measurements")

    tables = build_tables(
        processed, context["resolution"], context["resolution"].source_name,
        config, lidar_calibration, identity_mapping,
    )
    for name, value in tables.items():
        _write_public_table_json(version_dir / f"{name}.json", name, value)

    for item in processed:
        output_path = output_root / item["output_relative_path"]
        if os.environ.get("AGHRI_NUSCENES_METADATA_ONLY") == "1":
            item["output_sha256"] = hashlib.sha256(item["output_payload"]).hexdigest()
            item["output_byte_count"] = len(item["output_payload"])
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(item["output_payload"])
            item["output_sha256"] = sha256_file(output_path)
            item["output_byte_count"] = output_path.stat().st_size

    manifest_records = []
    for item in processed:
        boxes = []
        for box_item in item["boxes"]:
            box = box_item["source_box"]
            boxes.append({
                "source_record_index": box.record_index,
                "source_label_index": box.label_index,
                "private_source_identity": box.source_identity,
                "opaque_track": box_item["opaque_track"],
                "source_center_xyz": list(box.center),
                "source_extents_xyz": list(box.extents),
                "original_last_three_values": list(box.raw_rotation),
                "source_rotation_conversion_policy": "identity [1,0,0,0]",
                "num_lidar_pts": box_item["num_lidar_pts"],
                "sample_token": box_item["sample_token"],
                "sample_data_token": box_item["sample_data_token"],
                "instance_token": box_item["instance_token"],
                "annotation_token": box_item["annotation_token"],
                "global_translation": box_item["global_translation"],
                "global_rotation_wxyz": box_item["global_rotation_wxyz"],
                "output_size_wlh": box_item["output_size_wlh"],
            })
        interp = item["pose_interpolation"]
        manifest_records.append({
            "public_scene_name": context["resolution"].public_scene_name,
            "source_record_index": item["record_index"],
            "source_annotation_file_value": item["source_annotation_file_value"],
            "annotation_reference_repair": item["annotation_reference_repair"],
            "source_pcd": item["source_descriptor"],
            "source_timestamp_ns": item["timestamp_ns"],
            "serialized_timestamp": item["serialized_timestamp"],
            "serialized_vs_filename_delta_ns_approx": item["filename_timestamp_delta_ns_approx"],
            "output_timestamp_us": item["timestamp_us"],
            "discarded_submicrosecond_ns": item["discarded_submicrosecond_ns"],
            "pcd_header": item["pcd_header"],
            "point_conversion": {
                **item["point_stats"],
                "output_byte_count": item["output_byte_count"],
                "output_sha256": item["output_sha256"],
                "output_relative_path": item["output_relative_path"],
                "columns": ["x", "y", "z", "placeholder_zero", "placeholder_zero"],
            },
            "pose": {
                "source": "odom_global.jsonl",
                "left_index": interp.left_index,
                "right_index": interp.right_index,
                "left_time_s": interp.left_time_s,
                "right_time_s": interp.right_time_s,
                "amount": interp.amount,
                "bracket_ms": interp.bracket_ms,
                "tf_composition_translation_error_m": item["tf_composition_translation_error_m"],
                "tf_composition_rotation_error_rad": item["tf_composition_rotation_error_rad"],
                "rebased_translation": item["rebased_pose"].translation.tolist(),
                "rebased_rotation_wxyz": item["rebased_pose"].rotation_wxyz.tolist(),
            },
            "boxes": boxes,
        })
    _write_jsonl(manifest_dir / "conversion_manifest.jsonl", manifest_records)
    _write_jsonl(manifest_dir / "exclusions.jsonl", exclusions)
    _write_json(
        manifest_dir / "source_to_scene.json",
        {"mapping_policy": "train list, then val list, then test list; one-based numbering",
         "scenes": [resolution.__dict__ for resolution in resolutions]},
    )
    for descriptor, source_path in zip(used_sources, static_source_paths + [item["source_pcd"] for item in processed]):
        stat = source_path.stat()
        if (
            stat.st_size != descriptor["size_bytes"]
            or stat.st_mtime_ns != descriptor["mtime_ns"]
            or sha256_file(source_path) != descriptor["sha256"]
        ):
            raise ValueError(f"source file changed during conversion: {descriptor['relative_path']}")
    for descriptor in split_inputs:
        split_path = split_root / descriptor["relative_path"]
        stat = split_path.stat()
        if (
            stat.st_size != descriptor["size_bytes"]
            or stat.st_mtime_ns != descriptor["mtime_ns"]
            or sha256_file(split_path) != descriptor["sha256"]
        ):
            raise ValueError(f"split file changed during conversion: {descriptor['relative_path']}")
    conversion_config = {
        "converter_version": __version__,
        "configuration": config,
        "selected_annotation_reason": context["annotation_reason"],
        "calibration_evidence": calibration_evidence,
        "private_identity_mapping": identity_mapping,
        "source_files_used": used_sources,
        "split_files": split_inputs,
        "source_integrity_scope": (
            "Before/after size, mtime_ns, and SHA-256 for every source file used by "
            "the staging conversion (annotation, calibration, three pose/TF streams, "
            "and every included PCD), "
            "plus all three split lists. Unrelated source payloads were not scanned."
        ),
        "counts": {
            "annotation_records": len(records),
            "included_safe_pose_lidar_samples": len(processed),
            "excluded_lidar_measurements": sum(
                item.get("scope") == "lidar_measurement" for item in exclusions
            ),
            "excluded_malformed_boxes": sum(item.get("scope") == "box" for item in exclusions),
        },
        "source_root_written": False,
        "absolute_gps_emitted": False,
    }
    _write_json(manifest_dir / "conversion_config.json", conversion_config)
    visual_outputs = render_visual_checks(
        processed, report_dir, context["resolution"].public_scene_name
    )
    return {
        "processed": processed,
        "tables": tables,
        "manifest_records": manifest_records,
        "exclusions": exclusions,
        "visual_outputs": visual_outputs,
        "conversion_config": conversion_config,
    }
