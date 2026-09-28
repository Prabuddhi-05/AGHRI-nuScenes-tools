"""Materialize and merge all 65 AGHRI recording conversions."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import traceback

import yaml

from aghri_nuscenes.annotations import sha256_file
from aghri_nuscenes.camera_conversion import convert_camera_pilot
from aghri_nuscenes.camera_discovery import discover_camera_inputs
from aghri_nuscenes.converter import convert_pilot, _write_public_table_json
from aghri_nuscenes.discovery import build_pilot_plan, resolve_all_scenes


PUBLIC_TABLES = (
    "category", "attribute", "visibility", "sensor", "calibrated_sensor",
    "ego_pose", "log", "scene", "sample", "sample_data", "instance",
    "sample_annotation", "map", "splits", "camera_model",
)
CONSTANT_TABLES = (
    "category", "attribute", "visibility", "sensor", "calibrated_sensor",
    "map", "camera_model",
)
DYNAMIC_TABLES = (
    "ego_pose", "log", "scene", "sample", "sample_data", "instance",
    "sample_annotation",
)
CAMERAS = (
    "cam_zed_rgb", "cam_fish_front", "cam_fish_left", "cam_fish_right",
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=False, allow_nan=False) + "\n")


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))
            handle.write("\n")


def require_safe_output(output: Path, work: Path, protected: list[Path], resume: bool) -> None:
    resolved_output = output.resolve()
    resolved_work = work.resolve()
    if resolved_output == resolved_work or resolved_output in resolved_work.parents or resolved_work in resolved_output.parents:
        raise ValueError("output and work roots must be disjoint")
    for item in protected:
        root = item.resolve(strict=True)
        for candidate in (resolved_output, resolved_work):
            if candidate == root or root in candidate.parents or candidate in root.parents:
                raise ValueError("unsafe overlap with protected input: {}".format(root))
    for candidate in (output, work):
        if candidate.exists() and not candidate.is_dir():
            raise ValueError("output/work root is not a directory: {}".format(candidate))
        if not resume and candidate.exists() and any(candidate.iterdir()):
            raise ValueError("output/work root must be absent or empty: {}".format(candidate))


def _move_payload_tree(source_root: Path, output_root: Path) -> tuple[int, int]:
    moved = 0
    byte_count = 0
    for top in ("samples", "sweeps"):
        source = source_root / top
        if not source.is_dir():
            continue
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            relative = path.relative_to(source_root)
            destination = output_root / relative
            if destination.exists():
                raise ValueError("payload collision: {}".format(relative))
            destination.parent.mkdir(parents=True, exist_ok=True)
            size = path.stat().st_size
            path.replace(destination)
            moved += 1
            byte_count += size
    return moved, byte_count


def _load_scene_tables(camera_output: Path, version: str) -> dict:
    root = camera_output / version
    result = {}
    for name in PUBLIC_TABLES:
        path = root / "{}.json".format(name)
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = read_json(path)
    return result


def _table_hashes(version_dir: Path) -> dict[str, str]:
    return {name: sha256_file(version_dir / "{}.json".format(name)) for name in PUBLIC_TABLES}


def convert_full(source_root: Path, split_root: Path, output_root: Path, work_root: Path,
                 discovery_path: Path, lidar_config_path: Path, camera_config_path: Path,
                 overrides_path: Path, resume: bool = False,
                 protected_roots: tuple[Path, ...] = ()) -> dict:
    protected = [source_root, split_root, *protected_roots]
    require_safe_output(output_root, work_root, protected, resume)
    discovery = read_json(discovery_path)
    if discovery.get("decision") != "PASS" or discovery.get("resolved_scene_count") != 65:
        raise ValueError("the required conversion discovery gate has not passed")
    lidar_config = yaml.safe_load(lidar_config_path.read_text())
    camera_config = yaml.safe_load(camera_config_path.read_text())
    overrides = yaml.safe_load(overrides_path.read_text()) or {}
    resolutions, resolution_report = resolve_all_scenes(source_root, split_root)
    output_root.mkdir(parents=True, exist_ok=resume)
    work_root.mkdir(parents=True, exist_ok=resume)
    version_dir = output_root / camera_config["version"]
    reports_dir = output_root / "reports"
    manifests_dir = output_root / "manifests"
    synchronization_dir = output_root / "metadata" / "synchronization"
    if resume and version_dir.exists():
        raise ValueError("resume refused because final JSON version already exists: {}".format(version_dir))
    synchronization_dir.mkdir(parents=True, exist_ok=resume)
    (reports_dir / "per_scene_discovery").mkdir(parents=True, exist_ok=resume)
    (manifests_dir / "per_scene_completeness").mkdir(parents=True, exist_ok=resume)

    resume_count = 0
    prior_by_scene = {}
    if resume:
        progress_path = reports_dir / "conversion_progress.json"
        if not progress_path.is_file():
            raise ValueError("resume requested but progress file is missing: {}".format(progress_path))
        progress = read_json(progress_path)
        if bool(progress.get("metadata_only")) != (
            os.environ.get("AGHRI_NUSCENES_METADATA_ONLY") == "1"
        ):
            raise ValueError("resume mode differs from the recorded metadata_only mode")
        resume_count = int(progress.get("completed_scene_count", 0))
        if resume_count < 1 or resume_count >= len(resolutions):
            raise ValueError("invalid completed scene count for resume: {}".format(resume_count))
        previous = progress.get("per_scene", [])
        if len(previous) != resume_count:
            raise ValueError("progress per_scene length does not match completed count")
        for expected, saved in zip(resolutions[:resume_count], previous):
            if (saved.get("public_scene") != expected.public_scene_name or
                    saved.get("source_name") != expected.source_name or
                    saved.get("split") != expected.split):
                raise ValueError("progress scene order does not match authoritative resolution")
            prior_by_scene[expected.public_scene_name] = saved

    constant = {}
    dynamic = {name: [] for name in DYNAMIC_TABLES}
    all_lidar_manifest = []
    all_lidar_exclusions = []
    all_camera_dispositions = []
    all_camera_pose = []
    all_completeness = []
    per_scene = []
    sync_recordings = []
    payload_hashes = {}
    total_payload_files = 0
    total_payload_bytes = 0
    metadata_only = os.environ.get("AGHRI_NUSCENES_METADATA_ONLY") == "1"

    for scene_index, resolution in enumerate(resolutions, start=1):
        scene_work = work_root / resolution.public_scene_name
        lidar_output = scene_work / "lidar"
        camera_output = scene_work / "camera"
        resumed_scene = scene_index <= resume_count
        if resumed_scene:
            for required in (
                lidar_output / "reports/pilot_plan.json",
                camera_output / "reports/camera_discovery.json",
                camera_output / camera_config["version"],
            ):
                if not required.exists():
                    raise ValueError("completed resume scene is incomplete: {}".format(required))
            plan = read_json(lidar_output / "reports/pilot_plan.json")
            camera_discovery = read_json(camera_output / "reports/camera_discovery.json")
            context = {
                "records": [None] * int(prior_by_scene[resolution.public_scene_name]["annotation_records"]),
                "annotation_reference_corrections": [None] * int(
                    prior_by_scene[resolution.public_scene_name]["annotation_reference_repairs"]
                ),
            }
        else:
            if scene_work.exists() and any(scene_work.iterdir()):
                raise ValueError(
                    "unrecorded partial scene work blocks safe resume; move it aside first: {}".format(
                        scene_work
                    )
                )
            override = overrides.get("scenes", {}).get(resolution.source_name)
            plan, context = build_pilot_plan(
                source_root, split_root, lidar_output, resolution.source_name,
                resolutions, lidar_config, override,
            )
            convert_pilot(
                source_root, split_root, lidar_output, lidar_config, plan, context,
                resolutions, resolution_report,
            )
            camera_discovery, camera_context = discover_camera_inputs(
                source_root, lidar_output, resolution.source_name, camera_config
            )
            if camera_discovery["decision"] != "PASS":
                raise ValueError(
                    "camera discovery failed for {}: {}".format(
                        resolution.public_scene_name, camera_discovery["discovery_blockers"]
                    )
                )
            convert_camera_pilot(
                source_root, lidar_output, camera_output, resolution.source_name,
                camera_config, camera_discovery, camera_context,
            )
        scene_tables = _load_scene_tables(camera_output, camera_config["version"])
        for name in CONSTANT_TABLES:
            if name not in constant:
                constant[name] = scene_tables[name]
            elif scene_tables[name] != constant[name]:
                raise ValueError("constant table differs across scenes: {}".format(name))
        for name in DYNAMIC_TABLES:
            dynamic[name].extend(scene_tables[name])

        lidar_manifest = read_jsonl(lidar_output / "manifests/conversion_manifest.jsonl")
        lidar_exclusions = read_jsonl(lidar_output / "manifests/exclusions.jsonl")
        camera_dispositions = read_jsonl(camera_output / "manifests/camera_file_disposition.jsonl")
        camera_pose = read_jsonl(camera_output / "manifests/camera_ego_pose_manifest.jsonl")
        completeness = read_json(camera_output / "manifests/camera_completeness.json")
        sync = read_json(camera_output / "reports/sync/{}_sync.json".format(resolution.public_scene_name))
        for value in lidar_exclusions:
            value.setdefault("public_scene", resolution.public_scene_name)
            value.setdefault("source_name", resolution.source_name)
        for values in (camera_dispositions, camera_pose):
            for value in values:
                value["public_scene"] = resolution.public_scene_name
                value["source_name"] = resolution.source_name
        for value in completeness["samples"]:
            value["public_scene"] = resolution.public_scene_name
            value["source_name"] = resolution.source_name
            value["split"] = resolution.split
        all_lidar_manifest.extend(lidar_manifest)
        all_lidar_exclusions.extend(lidar_exclusions)
        all_camera_dispositions.extend(camera_dispositions)
        all_camera_pose.extend(camera_pose)
        all_completeness.extend(completeness["samples"])
        write_json(
            manifests_dir / "per_scene_completeness/{}.json".format(resolution.public_scene_name),
            completeness,
        )
        write_json(
            reports_dir / "per_scene_discovery/{}.json".format(resolution.public_scene_name),
            camera_discovery,
        )
        write_json(
            synchronization_dir / "{}_sync.json".format(resolution.public_scene_name), sync
        )
        sync_recordings.append({
            "recording": resolution.source_name,
            "public_scene": resolution.public_scene_name,
            "split": resolution.split,
            "chosen_threshold_s": sync["chosen_threshold_s"],
            "p95_preference_met": sync["p95_preference_met"],
            "estimated_camera_offsets_ns": sync["estimated_camera_offsets_ns"],
            "chosen_threshold_metrics": sync["chosen_threshold_metrics"],
        })
        for item in lidar_manifest:
            conversion = item["point_conversion"]
            payload_hashes[conversion["output_relative_path"]] = conversion["output_sha256"]
        for item in camera_dispositions:
            if item["disposition"] != "excluded":
                payload_hashes[item["output_relative_path"]] = item["output_sha256"]
        moved_files, moved_bytes = (0, 0)
        if resumed_scene:
            moved_files = int(prior_by_scene[resolution.public_scene_name]["moved_payload_files"])
            moved_bytes = int(prior_by_scene[resolution.public_scene_name]["moved_payload_bytes"])
        elif not metadata_only:
            moved_files, moved_bytes = _move_payload_tree(camera_output, output_root)
        total_payload_files += moved_files
        total_payload_bytes += moved_bytes
        role_counts = Counter(item["disposition"] for item in camera_dispositions)
        scene_summary = {
            "index": scene_index,
            "public_scene": resolution.public_scene_name,
            "source_name": resolution.source_name,
            "split": resolution.split,
            "annotation_records": len(context["records"]),
            "annotation_reference_repairs": len(context["annotation_reference_corrections"]),
            "included_lidar_samples": len(scene_tables["sample"]),
            "excluded_lidar_measurements": sum(
                item.get("scope") == "lidar_measurement" for item in lidar_exclusions
            ),
            "sample_annotations": len(scene_tables["sample_annotation"]),
            "instances": len(scene_tables["instance"]),
            "camera_source_files": sum(
                stream["file_count"] for stream in camera_discovery["camera_streams"].values()
            ),
            "camera_keyframes": role_counts["keyframe"],
            "camera_sweeps": role_counts["sweep"],
            "camera_exclusions": role_counts["excluded"],
            "complete_four_camera_samples": completeness["complete_four_camera_samples"],
            "incomplete_samples": completeness["incomplete_samples"],
            "chosen_threshold_s": sync["chosen_threshold_s"],
            "p95_preference_met": sync["p95_preference_met"],
            "moved_payload_files": moved_files,
            "moved_payload_bytes": moved_bytes,
        }
        per_scene.append(scene_summary)
        write_json(reports_dir / "conversion_progress.json", {
            "completed_scene_count": scene_index,
            "metadata_only": metadata_only,
            "last_scene": scene_summary,
            "per_scene": per_scene,
        })
        print(
            "{} {}/65 {} samples={} complete={} camera(k/s/x)={}/{}/{} threshold={:.2f}".format(
                "REUSE" if resumed_scene else "CONVERT",
                scene_index, resolution.public_scene_name, len(scene_tables["sample"]),
                completeness["complete_four_camera_samples"], role_counts["keyframe"],
                role_counts["sweep"], role_counts["excluded"], sync["chosen_threshold_s"],
            ),
            flush=True,
        )

    splits = {"aghri_train": [], "aghri_val": [], "aghri_test": []}
    for resolution in resolutions:
        splits["aghri_{}".format(resolution.split)].append(resolution.public_scene_name)
    final_tables = {**constant, **dynamic, "splits": splits}
    if set(final_tables) != set(PUBLIC_TABLES):
        raise ValueError("merged table set mismatch")
    version_dir.mkdir(parents=True)
    for name in PUBLIC_TABLES:
        _write_public_table_json(version_dir / "{}.json".format(name), name, final_tables[name])

    write_jsonl(manifests_dir / "conversion_manifest.jsonl", all_lidar_manifest)
    write_jsonl(manifests_dir / "exclusions.jsonl", all_lidar_exclusions)
    write_jsonl(manifests_dir / "camera_file_disposition.jsonl", all_camera_dispositions)
    write_jsonl(manifests_dir / "camera_ego_pose_manifest.jsonl", all_camera_pose)
    complete_count = sum(item["complete_four_camera_sample"] for item in all_completeness)
    write_json(manifests_dir / "camera_completeness.json", {
        "total_lidar_samples": len(all_completeness),
        "complete_four_camera_samples": complete_count,
        "incomplete_samples": len(all_completeness) - complete_count,
        "samples": all_completeness,
    })
    write_json(manifests_dir / "source_to_scene.json", resolution_report)
    write_json(reports_dir / "synchronization_summary.json", {
        "anchor": "lidar",
        "recording_count": len(sync_recordings),
        "fallback_recording_count": sum(not item["p95_preference_met"] for item in sync_recordings),
        "recordings": sync_recordings,
    })
    write_json(reports_dir / "per_scene_conversion.json", per_scene)
    write_json(reports_dir / "annotation_rotation_audit.json", {
        "source_rotation_policy": "identity for every source box",
        "original_values_preserved_in": "manifests/conversion_manifest.jsonl",
        "converted_annotation_count": len(final_tables["sample_annotation"]),
        "discovered_nonzero_source_rotation_boxes": discovery["nonzero_source_rotation_value_box_total"],
        "filename_reference_repairs": sum(item["annotation_reference_repairs"] for item in per_scene),
    })
    write_json(reports_dir / "pose_interpolation_audit.json", {
        "no_extrapolation": True,
        "maximum_allowed_bracket_ms": float(lidar_config["pose"]["maximum_bracket_ms"]),
        "included_lidar_pose_count": len(final_tables["sample"]),
        "excluded_unsafe_lidar_measurements": sum(item["excluded_lidar_measurements"] for item in per_scene),
        "included_camera_pose_count": len(all_camera_pose),
        "discovered_unsafe_camera_measurements": discovery["unsafe_camera_pose_total"],
        "details": "manifests/exclusions.jsonl and manifests/camera_file_disposition.jsonl",
    })
    payload_manifest = manifests_dir / "payload_sha256.txt"
    with payload_manifest.open("w", encoding="utf-8") as handle:
        for relative, digest in sorted(payload_hashes.items()):
            handle.write("{}  {}\n".format(digest, relative))
    summary = {
        "operation": "AGHRI-to-nuScenes full conversion materialization",
        "status": "MATERIALIZED_PENDING_VALIDATION",
        "metadata_only": metadata_only,
        "output_root": str(output_root),
        "work_root": str(work_root),
        "version": camera_config["version"],
        "scene_count": len(final_tables["scene"]),
        "split_scene_counts": {key: len(value) for key, value in splits.items()},
        "samples": len(final_tables["sample"]),
        "sample_data": len(final_tables["sample_data"]),
        "sample_annotations": len(final_tables["sample_annotation"]),
        "instances": len(final_tables["instance"]),
        "camera_keyframes": sum(item["camera_keyframes"] for item in per_scene),
        "camera_sweeps": sum(item["camera_sweeps"] for item in per_scene),
        "camera_exclusions": sum(item["camera_exclusions"] for item in per_scene),
        "complete_four_camera_samples": complete_count,
        "incomplete_samples": len(all_completeness) - complete_count,
        "lidar_pose_exclusions": sum(item["excluded_lidar_measurements"] for item in per_scene),
        "annotation_reference_repairs": sum(item["annotation_reference_repairs"] for item in per_scene),
        "payload_manifest_records": len(payload_hashes),
        "materialized_payload_files_moved": total_payload_files,
        "materialized_payload_bytes": total_payload_bytes,
        "public_table_sha256": _table_hashes(version_dir),
        "discovery_report_sha256": sha256_file(discovery_path),
        "images_modified": False,
        "zed_depth_included": False,
        "training_performed": False,
    }
    write_json(reports_dir / "conversion_summary.json", summary)
    write_json(manifests_dir / "conversion_config.json", {
        "summary": summary,
        "lidar_configuration": lidar_config,
        "camera_configuration": camera_config,
        "annotation_overrides": overrides,
        "source_tree_written": False,
        "protected_inputs_written": False,
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--discovery-report", type=Path, required=True)
    parser.add_argument("--lidar-config", type=Path, required=True)
    parser.add_argument("--camera-config", type=Path, required=True)
    parser.add_argument("--annotation-overrides", type=Path, required=True)
    parser.add_argument(
        "--protected-root",
        action="append",
        type=Path,
        default=[],
        help="additional read-only root that output/work directories must not overlap",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    source_root = args.source_root.resolve(strict=True)
    split_root = args.split_root.resolve(strict=True)
    output_root = args.output_root.resolve()
    work_root = args.work_root.resolve()
    protected_roots = tuple(path.resolve(strict=True) for path in args.protected_root)
    # Check safety before either execution or a failure report can write anywhere.
    require_safe_output(
        output_root,
        work_root,
        [source_root, split_root, *protected_roots],
        args.resume,
    )
    if not args.execute:
        print(json.dumps({
            "status": "DRY_RUN_ONLY",
            "discovery_status": read_json(args.discovery_report).get("decision"),
            "output_root": str(output_root),
            "work_root": str(work_root),
            "metadata_only": os.environ.get("AGHRI_NUSCENES_METADATA_ONLY") == "1",
            "resume": args.resume,
        }, indent=2))
        return
    try:
        summary = convert_full(
            source_root, split_root, output_root, work_root,
            args.discovery_report.resolve(strict=True), args.lidar_config.resolve(strict=True),
            args.camera_config.resolve(strict=True), args.annotation_overrides.resolve(strict=True),
            args.resume, protected_roots,
        )
    except Exception as error:
        failure = {
            "status": "FAILED_MATERIALIZATION",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        failure_path = output_root / "reports/conversion_failure.json"
        write_json(failure_path, failure)
        print(json.dumps(failure, indent=2), flush=True)
        raise SystemExit(2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
