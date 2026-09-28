"""Read-only split and source-scene discovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from .annotations import load_annotation_records, resolve_annotation_source, sha256_file, validate_box
from .pcd import parse_pcd_header
from .poses import load_pose_stream, resolve_lidar_calibration
from .timestamps import parse_pcd_filename


EXPECTED_SPLIT_COUNTS = {"train": 52, "val": 7, "test": 6}


@dataclass(frozen=True)
class SceneResolution:
    source_name: str
    source_relative_path: str
    public_scene_name: str
    split: str


def load_split_lists(split_root: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        path = split_root / f"{split}.txt"
        if not path.is_file():
            raise ValueError(f"missing split list: {path}")
        names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(names) != EXPECTED_SPLIT_COUNTS[split] or len(set(names)) != len(names):
            raise ValueError(
                f"invalid {split} split: expected {EXPECTED_SPLIT_COUNTS[split]} unique, got {len(names)}"
            )
        result[split] = names
    sets = {name: set(values) for name, values in result.items()}
    overlaps = {
        "train_val": sorted(sets["train"] & sets["val"]),
        "train_test": sorted(sets["train"] & sets["test"]),
        "val_test": sorted(sets["val"] & sets["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"split overlap detected: {overlaps}")
    if len(set().union(*sets.values())) != 65:
        raise ValueError("split union does not contain exactly 65 scenes")
    return result


def discover_scene_directories(source_root: Path) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for part in sorted(source_root.glob("dataset_part*")):
        if not part.is_dir():
            continue
        for path in sorted(part.iterdir()):
            if path.is_dir() and path.name.endswith("_label"):
                result.setdefault(path.name, []).append(path)
    return result


def resolve_all_scenes(source_root: Path, split_root: Path) -> tuple[list[SceneResolution], dict]:
    splits = load_split_lists(split_root)
    directories = discover_scene_directories(source_root)
    ordered = [(split, name) for split in ("train", "val", "test") for name in splits[split]]
    resolutions: list[SceneResolution] = []
    errors: list[dict] = []
    for index, (split, source_name) in enumerate(ordered, start=1):
        matches = directories.get(source_name, [])
        if len(matches) != 1:
            errors.append({"source_name": source_name, "matches": [str(p) for p in matches]})
            continue
        resolutions.append(
            SceneResolution(
                source_name=source_name,
                source_relative_path=matches[0].relative_to(source_root).as_posix(),
                public_scene_name=f"aghri-scene-{index:04d}",
                split=split,
            )
        )
    extra = sorted(set(directories) - {name for _, name in ordered})
    if errors or extra or len(resolutions) != 65:
        raise ValueError(f"scene resolution failed: errors={errors}, extra={extra}")
    report = {
        "split_counts": {split: len(names) for split, names in splits.items()},
        "total_unique_split_names": len({name for values in splits.values() for name in values}),
        "resolved_unique_scenes": len(resolutions),
        "overlap_count": 0,
        "unresolved": [],
        "extra_source_scenes": [],
        "scenes": [resolution.__dict__ for resolution in resolutions],
    }
    return resolutions, report


def describe_input_file(path: Path, relative_to: Path) -> dict:
    stat = path.stat()
    return {
        "relative_path": path.relative_to(relative_to).as_posix(),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
    }


def resolve_annotation_pcd_references(
    records: list[dict], available: dict[str, Path], repair_policy: dict | None
) -> tuple[list[dict], list[dict]]:
    """Resolve annotation File values, allowing only an explicit audited repair."""
    pcd_ns = {}
    for name in available:
        _, _, value = parse_pcd_filename(name)
        pcd_ns[name] = value
    output = []
    corrections = []
    used = set()
    for index, source_record in enumerate(records):
        record = dict(source_record)
        filename = record.get("File")
        if filename in available:
            selected = filename
        else:
            if not repair_policy:
                raise ValueError("annotation references missing PCD: {!r}".format(filename))
            if repair_policy.get("type") != "unique_nearest_pcd_by_decimal_timestamp":
                raise ValueError("unsupported annotation filename repair policy")
            suffix = str(repair_policy.get("source_suffix"))
            pattern = r"^(\d+)\.(\d{{1,9}}){}$".format(re.escape(suffix))
            match = re.fullmatch(pattern, str(filename))
            if not match:
                raise ValueError(
                    "annotation filename does not match approved repair form: {!r}".format(filename)
                )
            target_ns = (
                int(match.group(1)) * 1_000_000_000
                + int(match.group(2).ljust(9, "0"))
            )
            ranked = sorted(
                (abs(value - target_ns), value - target_ns, name)
                for name, value in pcd_ns.items() if name not in used
            )
            if not ranked:
                raise ValueError("no PCD remains for repaired annotation reference")
            best = ranked[0]
            maximum = int(repair_policy["maximum_absolute_delta_ns"])
            if best[0] > maximum:
                raise ValueError(
                    "nearest PCD exceeds approved repair tolerance: {} ns".format(best[0])
                )
            if len(ranked) > 1 and ranked[1][0] == best[0]:
                raise ValueError("annotation filename repair has an equal-distance tie")
            selected = best[2]
            correction = {
                "record_index": index,
                "source_file_value": filename,
                "selected_pcd": selected,
                "signed_delta_ns": int(best[1]),
                "absolute_delta_ns": int(best[0]),
                "next_nearest_absolute_delta_ns": int(ranked[1][0]) if len(ranked) > 1 else None,
                "policy": repair_policy["type"],
            }
            corrections.append(correction)
            record["_aghri_original_file_reference"] = filename
            record["_aghri_reference_repair"] = correction
            record["File"] = selected
        if selected in used:
            raise ValueError("annotation resolves one PCD more than once: {}".format(selected))
        used.add(selected)
        output.append(record)
    if repair_policy:
        expected = int(repair_policy["expected_repaired_records"])
        if len(corrections) != expected:
            raise ValueError(
                "expected {} repaired references, got {}".format(expected, len(corrections))
            )
        if repair_policy.get("require_all_pcds_covered") and used != set(available):
            raise ValueError("approved repair does not form complete annotation/PCD closure")
    return output, corrections


def build_pilot_plan(
    source_root: Path,
    split_root: Path,
    output_root: Path,
    scenario: str,
    resolutions: list[SceneResolution],
    config: dict,
    annotation_override: dict | None,
) -> tuple[dict, dict]:
    matches = [resolution for resolution in resolutions if resolution.source_name == scenario]
    if len(matches) != 1:
        raise ValueError(f"scenario must resolve once in split mapping: {scenario}")
    resolution = matches[0]
    scene_path = source_root / resolution.source_relative_path
    annotation_path, reason, candidates = resolve_annotation_source(
        scene_path, scenario, annotation_override
    )
    source_records = load_annotation_records(annotation_path)
    lidar_dir = scene_path / "sensor_data" / "lidar"
    available = {path.name: path for path in lidar_dir.glob("*.pcd")}
    repair_policy = (annotation_override or {}).get("filename_reference_repair")
    records, reference_corrections = resolve_annotation_pcd_references(
        source_records, available, repair_policy
    )
    referenced_names: list[str] = []
    missing: list[str] = []
    duplicate_references: list[str] = []
    seen: set[str] = set()
    valid_boxes = 0
    malformed_boxes = 0
    nonzero_rotations = 0
    timestamp_ns: list[int] = []
    for record_index, record in enumerate(records):
        if not isinstance(record, dict) or not isinstance(record.get("File"), str):
            raise ValueError(f"invalid annotation record at index {record_index}")
        filename = record["File"]
        _, _, exact_ns = parse_pcd_filename(filename)
        timestamp_ns.append(exact_ns)
        referenced_names.append(filename)
        if filename not in available:
            missing.append(filename)
        if filename in seen:
            duplicate_references.append(filename)
        seen.add(filename)
        labels = record.get("Labels", [])
        if not isinstance(labels, list):
            raise ValueError(f"Labels must be a list at record {record_index}")
        source_ids: set[str] = set()
        for label_index, label in enumerate(labels):
            box, exclusion = validate_box(record_index, label_index, label)
            if exclusion:
                malformed_boxes += 1
            else:
                assert box is not None
                if box.source_identity in source_ids:
                    raise ValueError(
                        f"duplicate source identity {box.source_identity!r} in record {record_index}"
                    )
                source_ids.add(box.source_identity)
                valid_boxes += 1
                nonzero_rotations += int(any(value != 0.0 for value in box.raw_rotation))
    if timestamp_ns != sorted(timestamp_ns) or len(timestamp_ns) != len(set(timestamp_ns)):
        raise ValueError("annotation filename timestamps are not strictly increasing and unique")
    orphan_names = sorted(set(available) - set(referenced_names))
    if missing or duplicate_references:
        raise ValueError(
            f"annotation/PCD integrity failure: missing={missing}, duplicate={duplicate_references}"
        )
    for filename in referenced_names:
        parse_pcd_header(available[filename])
    pose_cfg = config["pose"]
    global_stream = load_pose_stream(scene_path / pose_cfg["source"], "p", "odom_global")
    calibration_path = source_root / config["calibration"]["source"]
    _, calibration_evidence = resolve_lidar_calibration(calibration_path, config["calibration"])
    plan = {
        "mode": "dry-run pilot plan",
        "scenario": scenario,
        "resolved_source_scene": resolution.source_relative_path,
        "public_scene_name": resolution.public_scene_name,
        "authoritative_split": f"aghri_{resolution.split}",
        "selected_annotation": annotation_path.relative_to(source_root).as_posix(),
        "annotation_selection_reason": reason,
        "annotation_candidates": candidates,
        "annotation_records": len(records),
        "annotation_filename_reference_repairs": reference_corrections,
        "valid_source_boxes": valid_boxes,
        "malformed_source_boxes": malformed_boxes,
        "source_boxes_with_nonzero_rotation_values": nonzero_rotations,
        "referenced_pcd_files": len(referenced_names),
        "missing_referenced_pcd_files": len(missing),
        "orphan_pcd_files": len(orphan_names),
        "orphan_pcd_names": orphan_names,
        "pose_stream": {
            "available": True,
            "records": len(global_stream.times_s),
            "range_seconds": list(global_stream.time_range_s),
            "candidate_lidar_range_ns": [timestamp_ns[0], timestamp_ns[-1]],
        },
        "lidar_calibration": calibration_evidence,
        "output_path": str(output_root),
        "expected_payload_files": len(records),
        "expected_ego_pose_records": len(records),
        "expected_sample_records": len(records),
        "expected_sample_data_records": len(records),
        "expected_annotation_records": valid_boxes,
        "expected_json_files": 14,
    }
    context = {
        "resolution": resolution,
        "scene_path": scene_path,
        "annotation_path": annotation_path,
        "annotation_reason": reason,
        "annotation_reference_corrections": reference_corrections,
        "records": records,
        "available_pcd": available,
        "timestamp_ns": timestamp_ns,
    }
    return plan, context
