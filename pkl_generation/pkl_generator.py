"""Generate deterministic legacy MMDetection3D/BEVFusion PKL variants."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import pickle
import statistics
from typing import Any

import numpy as np

from pkl_annotations import (
    build_annotation_tracks,
    build_sample_annotation_arrays,
    estimate_track_velocities,
)
from pkl_contract import (
    CAMERA_VARIANTS,
    DATASET_VERSION,
    DETECTION_CLASS,
    LIDAR_CHANNEL,
    PKL_PROTOCOL,
    POINT_LOAD_DIM,
    SPLIT_KEY_TO_NAME,
    MAX_ALLOWED_SYNC_THRESHOLD_S,
    camera_channels,
    metadata,
    output_filename,
)
from pkl_splits import load_dataset_splits
from pkl_transforms import invert_transform, make_transform


TABLE_NAMES = (
    "sensor",
    "calibrated_sensor",
    "ego_pose",
    "log",
    "scene",
    "sample",
    "sample_data",
    "sample_annotation",
    "instance",
    "camera_model",
    "map",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            indent=2,
            sort_keys=False,
            allow_nan=False,
            default=json_default,
        )
        handle.write("\n")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_context(
    dataset_root: Path,
    synchronization_dir: Path | None = None,
) -> dict:
    dataset_root = dataset_root.resolve()
    version_dir = dataset_root / DATASET_VERSION
    if not version_dir.is_dir():
        raise FileNotFoundError(version_dir)
    tables = {name: load_json(version_dir / f"{name}.json") for name in TABLE_NAMES}
    if tables["map"] != []:
        raise ValueError("converted dataset map table is not empty")
    sensor_by_token = {item["token"]: item for item in tables["sensor"]}
    calibration_by_token = {item["token"]: item for item in tables["calibrated_sensor"]}
    pose_by_token = {item["token"]: item for item in tables["ego_pose"]}
    sample_by_token = {item["token"]: item for item in tables["sample"]}
    scene_by_token = {item["token"]: item for item in tables["scene"]}
    log_by_token = {item["token"]: item for item in tables["log"]}
    model_by_calibration = {
        item["calibrated_sensor_token"]: item for item in tables["camera_model"]
    }

    def channel_of(item: dict) -> str:
        calibration = calibration_by_token[item["calibrated_sensor_token"]]
        return sensor_by_token[calibration["sensor_token"]]["channel"]

    keyframes: dict[str, dict[str, dict]] = defaultdict(dict)
    nonkey_lidar = []
    for item in tables["sample_data"]:
        channel = channel_of(item)
        if channel == LIDAR_CHANNEL and not item["is_key_frame"]:
            nonkey_lidar.append(item["token"])
        if item["is_key_frame"]:
            if channel in keyframes[item["sample_token"]]:
                raise ValueError(f"duplicate keyframe for {item['sample_token']} / {channel}")
            keyframes[item["sample_token"]][channel] = item
    if nonkey_lidar:
        raise ValueError(f"unexpected genuine/non-key LiDAR records: {nonkey_lidar[:5]}")

    annotations_by_sample: dict[str, list[dict]] = defaultdict(list)
    for item in tables["sample_annotation"]:
        annotations_by_sample[item["sample_token"]].append(item)

    membership, raw_splits = load_dataset_splits(version_dir)
    expected_split_scene_counts = {"aghri_train": 52, "aghri_val": 7, "aghri_test": 6}
    actual_split_scene_counts = {key: len(value) for key, value in raw_splits.items()}
    if actual_split_scene_counts != expected_split_scene_counts:
        raise ValueError(f"dataset split counts differ from 52/7/6: {actual_split_scene_counts}")
    public_scenes = {scene["name"] for scene in tables["scene"]}
    if set(membership) != public_scenes or len(public_scenes) != 65:
        raise ValueError("scene table and splits.json do not close over exactly 65 scenes")

    samples_by_scene: dict[str, list[dict]] = defaultdict(list)
    for sample in tables["sample"]:
        samples_by_scene[sample["scene_token"]].append(sample)
    sync_by_sample = {}
    sync_by_scene = {}
    source_recording_by_scene = {}
    if synchronization_dir is None:
        synchronization_dir = dataset_root / "metadata" / "synchronization"
    synchronization_dir = synchronization_dir.resolve()
    if not synchronization_dir.is_dir():
        raise FileNotFoundError(synchronization_dir)
    for scene in tables["scene"]:
        scene_name = scene["name"]
        sync_path = synchronization_dir / f"{scene_name}_sync.json"
        if not sync_path.is_file():
            raise FileNotFoundError(sync_path)
        sync = load_json(sync_path)
        if sync.get("public_scene") != scene_name:
            raise ValueError(f"sync public scene mismatch: {sync_path}")
        selected_threshold = float(sync["chosen_threshold_s"])
        if not (0.10 <= selected_threshold <= MAX_ALLOWED_SYNC_THRESHOLD_S):
            raise ValueError(f"invalid selected sync threshold for {scene_name}: {selected_threshold}")
        rows = {int(item["lidar_index"]): item for item in sync["samples"]}
        scene_samples = sorted(
            samples_by_scene[scene["token"]], key=lambda item: (int(item["timestamp"]), item["token"])
        )
        if set(rows) != set(range(len(scene_samples))):
            raise ValueError(f"synchronization rows do not cover {scene_name}")
        for local_index, sample in enumerate(scene_samples):
            row = rows[local_index]
            if int(row["lidar_timestamp_ns"]) // 1000 != int(sample["timestamp"]):
                raise ValueError(f"synchronization/sample timestamp mismatch in {scene_name}")
            sync_by_sample[sample["token"]] = {
                "recording": sync,
                "sample": row,
            }
        sync_by_scene[scene_name] = sync
        source_recording_by_scene[scene_name] = sync["recording"]
    return {
        "tables": tables,
        "sensor_by_token": sensor_by_token,
        "calibration_by_token": calibration_by_token,
        "pose_by_token": pose_by_token,
        "sample_by_token": sample_by_token,
        "scene_by_token": scene_by_token,
        "log_by_token": log_by_token,
        "model_by_calibration": model_by_calibration,
        "keyframes": dict(keyframes),
        "annotations_by_sample": dict(annotations_by_sample),
        "sync_by_sample": sync_by_sample,
        "sync_by_scene": sync_by_scene,
        "source_recording_by_scene": source_recording_by_scene,
        "membership": membership,
        "raw_splits": raw_splits,
        "dataset_root": dataset_root,
        "version_dir": version_dir,
    }


def _quality_band(residual_s: float, selected_threshold_s: float) -> str:
    if residual_s <= 0.050:
        return "<= 50 ms"
    if residual_s <= 0.100:
        return "> 50 and <= 100 ms"
    if residual_s <= 0.200:
        return "> 100 and <= 200 ms"
    if residual_s <= selected_threshold_s:
        return "> 200 ms to selected threshold"
    raise ValueError(f"eligible residual exceeds selected threshold: {residual_s}")


def _distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean_s": None, "median_s": None, "p95_s": None,
                "p99_s": None, "maximum_s": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean_s": float(np.mean(array)),
        "median_s": float(np.median(array)),
        "p95_s": float(np.percentile(array, 95)),
        "p99_s": float(np.percentile(array, 99)),
        "maximum_s": float(np.max(array)),
    }


def _recording_for_scene(scene: dict, log_by_token: dict[str, dict]) -> str:
    logfile = log_by_token[scene["log_token"]]["logfile"]
    return Path(logfile).stem if logfile.endswith(".bag") else logfile


def _camera_record(
    channel: str,
    camera_sd: dict,
    camera_calibration: dict,
    camera_pose: dict,
    camera_model: dict,
    global_from_key_lidar: np.ndarray,
    sync_camera: dict,
    offset_ns: int,
    selected_threshold_s: float,
) -> dict:
    global_from_camera = make_transform(
        camera_pose["translation"], camera_pose["rotation"]
    ) @ make_transform(camera_calibration["translation"], camera_calibration["rotation"])
    key_lidar_from_camera = invert_transform(global_from_key_lidar) @ global_from_camera
    original_ns = int(sync_camera["original_timestamp_ns"])
    aligned_ns = int(sync_camera["aligned_timestamp_ns"])
    residual_s = int(sync_camera["absolute_aligned_residual_ns"]) / 1_000_000_000.0
    if camera_sd["timestamp"] != original_ns // 1000:
        raise ValueError(f"camera timestamp mismatch: {channel} / {camera_sd['token']}")
    if Path(camera_sd["filename"]).name.split("__", 1)[-1] != sync_camera["selected_filename"]:
        raise ValueError(f"camera synchronization filename mismatch: {channel}")
    return {
        "data_path": camera_sd["filename"],
        "type": channel,
        "sample_data_token": camera_sd["token"],
        "sensor2ego_translation": list(map(float, camera_calibration["translation"])),
        "sensor2ego_rotation": list(map(float, camera_calibration["rotation"])),
        "ego2global_translation": list(map(float, camera_pose["translation"])),
        "ego2global_rotation": list(map(float, camera_pose["rotation"])),
        "timestamp": int(camera_sd["timestamp"]),
        "sensor2lidar_rotation": key_lidar_from_camera[:3, :3].astype(np.float64),
        "sensor2lidar_translation": key_lidar_from_camera[:3, 3].astype(np.float64),
        "cam_intrinsic": np.asarray(camera_calibration["camera_intrinsic"], dtype=np.float64),
        "camera_model_token": camera_model["token"],
        "camera_model": camera_model["camera_model"],
        "distortion_model": camera_model["distortion_model"],
        "distortion_coefficients": np.asarray(
            camera_model["distortion_coefficients"], dtype=np.float64
        ),
        "distortion_coefficient_order": list(camera_model["coefficient_order"]),
        "image_state": camera_model["image_state"],
        "image_width": int(camera_model["image_width"]),
        "image_height": int(camera_model["image_height"]),
        "calibration_source": camera_model.get("calibration_source"),
        "rectification_matrix": np.asarray(camera_model["rectification_matrix"], dtype=np.float64),
        "projection_matrix": np.asarray(camera_model["projection_matrix"], dtype=np.float64),
        "time_offset_s": offset_ns / 1_000_000_000.0,
        "original_timestamp_ns": original_ns,
        "aligned_timestamp": aligned_ns // 1000,
        "aligned_timestamp_ns": aligned_ns,
        "sync_residual_s": residual_s,
        "chosen_threshold_s": selected_threshold_s,
        "selected_keyframe_association": True,
    }


def build_all(
    variant: str,
    dataset_root: Path,
    synchronization_dir: Path | None = None,
) -> dict:
    selected_cameras = camera_channels(variant)
    context = load_context(dataset_root, synchronization_dir)
    tables = context["tables"]
    tracks = build_annotation_tracks(tables["sample_annotation"], context["sample_by_token"])
    velocity_by_annotation = estimate_track_velocities(tracks)
    split_infos: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    eligibility_rows = []
    sync_rows = []
    velocity_rows = []
    referenced_payloads: set[str] = set()
    complete_tokens = []
    le100_tokens = []
    le200_tokens = []

    sortable = []
    for original_index, sample in enumerate(tables["sample"]):
        scene = context["scene_by_token"][sample["scene_token"]]
        sortable.append((scene["name"], scene["token"], int(sample["timestamp"]), sample["token"], original_index, sample))

    for _, _, _, _, original_index, sample in sorted(sortable):
        scene = context["scene_by_token"][sample["scene_token"]]
        scene_name = scene["name"]
        if scene_name not in context["membership"]:
            raise ValueError(f"converted scene missing split: {scene_name}")
        split = context["membership"][scene_name]
        available = context["keyframes"].get(sample["token"], {})
        missing_lidar = LIDAR_CHANNEL not in available
        missing_cameras = [
            channel for channel in selected_cameras if channel not in available
        ]
        missing_required = ([LIDAR_CHANNEL] if missing_lidar else []) + missing_cameras
        eligible = not missing_required
        eligibility = {
            "sample_index_in_source_dataset": original_index,
            "sample_token": sample["token"],
            "scene": scene_name,
            "split": split,
            "timestamp": int(sample["timestamp"]),
            "sensor_configuration": variant,
            "required_camera_channels": list(selected_cameras),
            "complete_required_cameras": not missing_cameras,
            "missing_required_channels": missing_required,
            "disposition": "included" if eligible else "excluded_from_variant_pkls",
            "reason": None if eligible else "missing_required_keyframe",
        }
        # Retain the historical field only for the all-four-camera variant.
        if variant == "full":
            eligibility["complete_four_camera"] = eligible
        eligibility_rows.append(eligibility)
        if not eligible:
            continue

        lidar_sd = available.get(LIDAR_CHANNEL)
        if lidar_sd is None:
            raise ValueError(f"complete camera sample has no LiDAR: {sample['token']}")
        lidar_calibration = context["calibration_by_token"][lidar_sd["calibrated_sensor_token"]]
        lidar_pose = context["pose_by_token"][lidar_sd["ego_pose_token"]]
        global_from_key_lidar = make_transform(
            lidar_pose["translation"], lidar_pose["rotation"]
        ) @ make_transform(lidar_calibration["translation"], lidar_calibration["rotation"])
        lidar_from_global = invert_transform(global_from_key_lidar)
        sync_bundle = context["sync_by_sample"][sample["token"]]
        sync = sync_bundle["recording"]
        sync_sample = sync_bundle["sample"]
        selected_threshold_s = float(sync["chosen_threshold_s"])
        if int(sync_sample["lidar_timestamp_ns"]) // 1000 != int(lidar_sd["timestamp"]):
            raise ValueError("LiDAR synchronization order/timestamp mismatch")
        if Path(lidar_sd["filename"]).name.split("__", 1)[-1].removesuffix(".bin") != sync_sample["lidar_filename"]:
            raise ValueError("LiDAR synchronization filename mismatch")

        cameras = {}
        residuals = []
        for channel in selected_cameras:
            camera_sd = available[channel]
            camera_calibration = context["calibration_by_token"][
                camera_sd["calibrated_sensor_token"]
            ]
            camera_pose = context["pose_by_token"][camera_sd["ego_pose_token"]]
            camera_model = context["model_by_calibration"][camera_calibration["token"]]
            sync_camera = sync_sample["cameras"][channel]
            if sync_camera["rejection_reason"] is not None:
                raise ValueError("eligible camera keyframe was rejected by the stored synchronization metadata")
            offset_ns = int(sync["estimated_camera_offsets_ns"][channel])
            record = _camera_record(
                channel,
                camera_sd,
                camera_calibration,
                camera_pose,
                camera_model,
                global_from_key_lidar,
                sync_camera,
                offset_ns,
                selected_threshold_s,
            )
            cameras[channel] = record
            residuals.append(record["sync_residual_s"])
            referenced_payloads.add(camera_sd["filename"])
            sync_rows.append(
                {
                    "scene": scene_name,
                    "split": split,
                    "sample_token": sample["token"],
                    "lidar_timestamp": int(lidar_sd["timestamp"]),
                    "lidar_timestamp_ns": int(sync_sample["lidar_timestamp_ns"]),
                    "camera_channel": channel,
                    "camera_token": camera_sd["token"],
                    "camera_path": camera_sd["filename"],
                    "camera_original_timestamp": int(camera_sd["timestamp"]),
                    "camera_original_timestamp_ns": record["original_timestamp_ns"],
                    "camera_aligned_timestamp": record["aligned_timestamp"],
                    "camera_aligned_timestamp_ns": record["aligned_timestamp_ns"],
                    "offset_s": record["time_offset_s"],
                    "absolute_residual_s": record["sync_residual_s"],
                    "threshold_s": record["chosen_threshold_s"],
                    "quality_band": _quality_band(
                        record["sync_residual_s"], selected_threshold_s
                    ),
                }
            )

        annotations = context["annotations_by_sample"].get(sample["token"], [])
        arrays, annotation_velocity_rows = build_sample_annotation_arrays(
            annotations, lidar_from_global, velocity_by_annotation
        )
        for row in annotation_velocity_rows:
            row.update({"scene": scene_name, "split": split, "sample_token": sample["token"]})
        velocity_rows.extend(annotation_velocity_rows)
        referenced_payloads.add(lidar_sd["filename"])
        max_residual = max(residuals)
        info = {
            "lidar_path": lidar_sd["filename"],
            "token": sample["token"],
            "sweeps": [],
            "cams": cameras,
            "lidar2ego_translation": list(map(float, lidar_calibration["translation"])),
            "lidar2ego_rotation": list(map(float, lidar_calibration["rotation"])),
            "ego2global_translation": list(map(float, lidar_pose["translation"])),
            "ego2global_rotation": list(map(float, lidar_pose["rotation"])),
            "timestamp": int(lidar_sd["timestamp"]),
            "gt_boxes": arrays["gt_boxes"],
            "gt_names": arrays["gt_names"],
            "gt_velocity": arrays["gt_velocity"],
            "num_lidar_pts": arrays["num_lidar_pts"],
            "num_radar_pts": arrays["num_radar_pts"],
            "valid_flag": arrays["valid_flag"],
            "num_features": POINT_LOAD_DIM,
            "scene_name": scene_name,
            "source_scene_token": scene["token"],
            "aghri_sync": {
                "required_camera_channels": list(selected_cameras),
                "complete_required_cameras": True,
                "max_residual_s": float(max_residual),
                "mean_residual_s": float(statistics.fmean(residuals)),
                "all_within_050ms": bool(max_residual <= 0.050),
                "all_within_100ms": bool(max_residual <= 0.100),
                "all_within_200ms": bool(max_residual <= 0.200),
                "chosen_threshold_s": selected_threshold_s,
            },
        }
        if variant == "full":
            info["aghri_sync"]["complete_four_camera"] = True
        split_infos[split].append(info)
        complete_tokens.append(sample["token"])
        if max_residual <= 0.100:
            le100_tokens.append(sample["token"])
        if max_residual <= 0.200:
            le200_tokens.append(sample["token"])

    full_split_rows = []
    for split_key, scene_names in context["raw_splits"].items():
        split = SPLIT_KEY_TO_NAME[split_key]
        for scene_name in scene_names:
            full_split_rows.append({
                "recording": context["source_recording_by_scene"][scene_name],
                "public_scene": scene_name,
                "split": split,
                "status": "converted_source_scene",
            })
    converted_recordings = set(context["source_recording_by_scene"].values())
    return {
        "context": context,
        "split_infos": split_infos,
        "eligibility_rows": eligibility_rows,
        "sync_rows": sync_rows,
        "velocity_rows": velocity_rows,
        "referenced_payloads": sorted(referenced_payloads),
        "quality_lists": {
            "complete_all": complete_tokens,
            "complete_max_residual_le_100ms": le100_tokens,
            "complete_max_residual_le_200ms": le200_tokens,
        },
        "full_split_rows": full_split_rows,
        "converted_recordings": sorted(converted_recordings),
        "variant": variant,
        "camera_channels": list(selected_cameras),
    }


def _safe_output(output_dir: Path, dataset_root: Path) -> None:
    output_resolved = output_dir.resolve()
    input_resolved = dataset_root.resolve()
    if output_resolved == input_resolved or input_resolved in output_resolved.parents:
        raise ValueError("unsafe output overlap with converted dataset input")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"output directory must be empty: {output_dir}")


def _sync_summary(
    rows: list[dict],
    quality_lists: dict[str, list[str]],
    selected_cameras: tuple[str, ...],
) -> dict:
    bands = [
        "<= 50 ms",
        "> 50 and <= 100 ms",
        "> 100 and <= 200 ms",
        "> 200 ms to selected threshold",
    ]
    per_camera_split = {}
    for split in ("train", "val", "test"):
        per_camera_split[split] = {}
        for channel in selected_cameras:
            selected = [
                row for row in rows
                if row["split"] == split and row["camera_channel"] == channel
            ]
            counts = Counter(row["quality_band"] for row in selected)
            distribution = _distribution([row["absolute_residual_s"] for row in selected])
            distribution["bands"] = {
                band: {
                    "count": counts.get(band, 0),
                    "percentage": (100.0 * counts.get(band, 0) / len(selected)) if selected else 0.0,
                }
                for band in bands
            }
            per_camera_split[split][channel] = distribution
    per_scene = {}
    for scene_name in sorted({row["scene"] for row in rows}):
        selected = [row for row in rows if row["scene"] == scene_name]
        per_scene[scene_name] = {
            "split": selected[0]["split"],
            "selected_threshold_s": max(row["threshold_s"] for row in selected),
            **_distribution([row["absolute_residual_s"] for row in selected]),
        }
    fallback_scenes = sorted(
        scene for scene, values in per_scene.items()
        if values["selected_threshold_s"] > 0.10
    )
    return {
        "warning": (
            "Synchronization thresholds are selected per recording. Values above 0.10 s "
            "are fallback acceptance limits, not 50 ms accuracy claims."
        ),
        "eligible_sample_camera_rows": len(rows),
        "expected_rows": len(quality_lists["complete_all"]) * len(selected_cameras),
        "quality_list_counts": {key: len(value) for key, value in quality_lists.items()},
        "per_split_per_camera": per_camera_split,
        "per_scene": per_scene,
        "fallback_scene_count": len(fallback_scenes),
        "fallback_scenes": fallback_scenes,
        "worst_20_matches": sorted(
            rows, key=lambda row: row["absolute_residual_s"], reverse=True
        )[:20],
    }


def _eligibility_summary(built: dict) -> dict:
    rows = built["eligibility_rows"]
    included = [row for row in rows if row["disposition"] == "included"]
    excluded = [row for row in rows if row["disposition"] != "included"]
    source_scenes = {
        split: sorted(
            scene for scene, assigned_split in built["context"]["membership"].items()
            if assigned_split == split
        )
        for split in ("train", "val", "test")
    }
    included_scenes = {
        split: sorted({row["scene"] for row in included if row["split"] == split})
        for split in ("train", "val", "test")
    }
    exclusion_reasons = Counter(
        "+".join(row["missing_required_channels"]) for row in excluded
    )
    return {
        "sensor_configuration": built["variant"],
        "required_camera_channels": built["camera_channels"],
        "total_lidar_anchored_samples_examined": len(rows),
        "included_samples_by_split": {
            split: sum(row["split"] == split for row in included)
            for split in ("train", "val", "test")
        },
        "excluded_samples_by_split": {
            split: sum(row["split"] == split for row in excluded)
            for split in ("train", "val", "test")
        },
        "exclusion_reasons": dict(sorted(exclusion_reasons.items())),
        "included_sample_tokens": [row["sample_token"] for row in included],
        "excluded_samples": [
            {
                "sample_token": row["sample_token"],
                "scene": row["scene"],
                "split": row["split"],
                "missing_required_channels": row["missing_required_channels"],
                "reason": row["reason"],
            }
            for row in excluded
        ],
        "source_scenes_by_split": source_scenes,
        "source_scene_counts_by_split": {
            split: len(names) for split, names in source_scenes.items()
        },
        "included_scenes_by_split": included_scenes,
        "included_scene_counts_by_split": {
            split: len(names) for split, names in included_scenes.items()
        },
    }


def generate(
    output_dir: Path,
    reports_dir: Path,
    manifests_dir: Path,
    dry_run: bool,
    variant: str,
    dataset_root: Path,
    hash_payloads: bool = True,
    synchronization_dir: Path | None = None,
) -> dict:
    selected_cameras = camera_channels(variant)
    built = build_all(
        variant=variant,
        dataset_root=dataset_root,
        synchronization_dir=synchronization_dir,
    )
    counts = {split: len(infos) for split, infos in built["split_infos"].items()}
    raw_count = len(built["eligibility_rows"])
    incomplete_count = sum(
        row["disposition"] != "included" for row in built["eligibility_rows"]
    )
    if sum(counts.values()) + incomplete_count != raw_count:
        raise ValueError("eligible and excluded samples do not close over source samples")
    split_scene_counts = Counter(built["context"]["membership"].values())
    if dict(split_scene_counts) != {"train": 52, "val": 7, "test": 6}:
        raise ValueError(f"unexpected source split scene counts: {dict(split_scene_counts)}")
    preview = {
        "dry_run": dry_run,
        "sensor_configuration": variant,
        "camera_channels": list(selected_cameras),
        "output_filenames": {
            split: output_filename(variant, split)
            for split in ("train", "val", "test")
        },
        "source_sample_count": raw_count,
        "split_record_counts": counts,
        "eligible_total": sum(counts.values()),
        "incomplete_total": incomplete_count,
        "retained_percentage": 100.0 * sum(counts.values()) / raw_count,
        "referenced_payload_count": len(built["referenced_payloads"]),
        "payload_hashing_requested": bool(hash_payloads and not dry_run),
        "annotation_count": sum(len(info["gt_boxes"]) for infos in built["split_infos"].values() for info in infos),
    }
    eligibility_summary = _eligibility_summary(built)
    if dry_run:
        reports_dir.mkdir(parents=True, exist_ok=True)
        manifests_dir.mkdir(parents=True, exist_ok=True)
        write_json(reports_dir / "pkl_generation_dry_run_summary.json", preview)
        write_json(reports_dir / "eligibility_dry_run.json", eligibility_summary)
        write_json(manifests_dir / "sample_eligibility_dry_run.json", built["eligibility_rows"])
        return preview

    _safe_output(output_dir, dataset_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    pkl_files = {}
    for split in ("train", "val", "test"):
        payload = {
            "infos": built["split_infos"][split],
            "metadata": metadata(variant=variant),
        }
        path = output_dir / output_filename(variant, split)
        with path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=PKL_PROTOCOL)
        pkl_files[split] = {
            "path": str(path),
            "records": len(payload["infos"]),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    write_json(reports_dir / "sample_eligibility_manifest.json", built["eligibility_rows"])
    write_json(reports_dir / "eligibility_summary.json", eligibility_summary)
    excluded = [
        row for row in built["eligibility_rows"] if row["disposition"] != "included"
    ]
    write_json(manifests_dir / "incomplete_sample_exclusion.json", excluded)
    write_json(reports_dir / "sync_quality_manifest.json", built["sync_rows"])
    write_json(
        reports_dir / "sync_quality_summary.json",
        _sync_summary(built["sync_rows"], built["quality_lists"], selected_cameras),
    )
    write_json(reports_dir / "quality_token_lists.json", built["quality_lists"])
    write_json(reports_dir / "velocity_manifest.json", built["velocity_rows"])
    write_json(
        reports_dir / "full_split_manifest_status.json",
        {
            "rows": built["full_split_rows"],
            "status_counts": dict(Counter(row["status"] for row in built["full_split_rows"])),
            "converted_recordings": built["converted_recordings"],
        },
    )

    payload_hash_path = None
    if hash_payloads:
        payload_hash_path = manifests_dir / "referenced_payloads_before.sha256"
        with payload_hash_path.open("w", encoding="utf-8") as handle:
            for relative in built["referenced_payloads"]:
                absolute = dataset_root / relative
                if not absolute.is_file():
                    raise FileNotFoundError(absolute)
                handle.write(f"{sha256_file(absolute)}  {relative}\n")
    else:
        # This records what the PKLs reference without rereading every payload byte.
        # PKL contents are identical whether payload hashing is enabled or skipped.
        write_json(
            manifests_dir / "referenced_payload_paths.json",
            built["referenced_payloads"],
        )

    methods = Counter(row["method"] for row in built["velocity_rows"])
    finite_count = sum(row["finite_estimate"] for row in built["velocity_rows"])
    summary = {
        **preview,
        "dry_run": False,
        "pkl_files": pkl_files,
        "source_scene_count": len(built["context"]["tables"]["scene"]),
        "source_scenes": [item["name"] for item in built["context"]["tables"]["scene"]],
        "split_scene_counts": dict(split_scene_counts),
        "split_source": f"{DATASET_VERSION}/splits.json",
        "camera_order": list(selected_cameras),
        "lidar_channel": LIDAR_CHANNEL,
        "point_load_dim": POINT_LOAD_DIM,
        "point_use_dim": POINT_LOAD_DIM,
        "detection_class": DETECTION_CLASS,
        "sweeps_policy": "all empty; converted dataset has no validated non-keyframe LiDAR",
        "database_sampler": "disabled; no database generated",
        "payload_hash_manifest": (
            str(payload_hash_path) if payload_hash_path is not None else None
        ),
        "test_ground_truth_policy": "retained for internal evaluation only; never training input",
        "velocity": {
            "total_annotations": len(built["velocity_rows"]),
            "finite_estimates": finite_count,
            "explicit_missing_nan_fallbacks": len(built["velocity_rows"]) - finite_count,
            "methods": dict(methods),
        },
        "sync_warning": (
            "Per-recording selected thresholds are copied from conversion metadata; fallback limits "
            "must not be interpreted as 50 ms synchronization accuracy."
        ),
    }
    write_json(reports_dir / "pkl_generation_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--synchronization-dir",
        type=Path,
        help=(
            "directory containing aghri-scene-XXXX_sync.json; defaults to "
            "DATASET_ROOT/metadata/synchronization"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--report-dir", "--reports-dir", dest="reports_dir", type=Path,
        required=True,
    )
    parser.add_argument(
        "--manifest-dir", "--manifests-dir", dest="manifests_dir", type=Path,
        required=True,
    )
    parser.add_argument("--variant", choices=tuple(CAMERA_VARIANTS), default="full")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-payload-hashes",
        action="store_true",
        help=(
            "do not reread and SHA-256 every referenced image/LiDAR payload; "
            "this does not change the generated PKLs"
        ),
    )
    args = parser.parse_args()
    result = generate(
        args.output_dir,
        args.reports_dir,
        args.manifests_dir,
        args.dry_run,
        variant=args.variant,
        dataset_root=args.dataset_root,
        hash_payloads=not args.skip_payload_hashes,
        synchronization_dir=args.synchronization_dir,
    )
    print(json.dumps(result, indent=2, default=json_default, allow_nan=False))


if __name__ == "__main__":
    main()
