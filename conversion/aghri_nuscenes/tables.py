"""Construction of the Stage-1 raw nuScenes JSON tables."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
import re
from typing import Any

import numpy as np

from .tokens import deterministic_token
from .transforms import matrix_to_quaternion, quaternion_multiply, transform_point


def _link(records: list[dict], groups: list[list[int]]) -> None:
    for indices in groups:
        for position, record_index in enumerate(indices):
            records[record_index]["prev"] = records[indices[position - 1]]["token"] if position else ""
            records[record_index]["next"] = (
                records[indices[position + 1]]["token"] if position + 1 < len(indices) else ""
            )


def _date_from_scenario(scenario: str) -> str:
    matches = re.findall(r"_(\d{2})_(\d{2})_(\d{4})(?:_|$)", scenario)
    if len(matches) != 1:
        raise ValueError(f"could not extract one capture date from {scenario}")
    month, day, year = map(int, matches[0])
    return date(year, month, day).isoformat()


def _location_from_scenario(scenario: str, locations: dict[str, str]) -> tuple[str, str]:
    matches = [(prefix, value) for prefix, value in locations.items() if scenario.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"scenario has no unique configured location: {scenario}")
    return matches[0]


def build_tables(
    processed: list[dict[str, Any]],
    resolution,
    scenario: str,
    config: dict,
    lidar_calibration,
    identity_mapping: dict[str, str],
) -> dict[str, Any]:
    namespace = config["uuid_namespace"]
    token = lambda canonical: deterministic_token(namespace, canonical)
    public_scene = resolution.public_scene_name

    category_token = token("category|human")
    category = [{
        "token": category_token,
        "name": "human",
        "description": "Human participant represented by a 3D detection box.",
    }]
    attribute: list = []
    visibility: list = []

    sensor = []
    sensor_tokens: dict[str, str] = {}
    for definition in config["sensor_channels"]:
        channel = definition["channel"]
        sensor_token = token(f"sensor|{channel}")
        sensor_tokens[channel] = sensor_token
        sensor.append({"token": sensor_token, "channel": channel, "modality": definition["modality"]})

    calibration_token = token("calibrated_sensor|lidar|base_link|front_lidar_link")
    calibrated_sensor = [{
        "token": calibration_token,
        "sensor_token": sensor_tokens["lidar"],
        "translation": lidar_calibration.translation.astype(float).tolist(),
        "rotation": lidar_calibration.rotation_wxyz.astype(float).tolist(),
        "camera_intrinsic": [],
    }]

    location_prefix, location = _location_from_scenario(scenario, config["locations"])
    scene_number = public_scene.rsplit("-", 1)[-1]
    log_token = token(f"log|{public_scene}")
    log = [{
        "token": log_token,
        "logfile": f"aghri-recording-{scene_number}",
        "vehicle": "aghri_robot_01",
        "date_captured": _date_from_scenario(scenario),
        "location": location,
    }]

    sample: list[dict] = []
    sample_data: list[dict] = []
    ego_pose: list[dict] = []
    sample_annotation: list[dict] = []
    annotation_indices_by_track: dict[str, list[int]] = defaultdict(list)

    for item in processed:
        timestamp_us = item["timestamp_us"]
        sample_token = token(f"sample|{public_scene}|{timestamp_us}")
        data_token = token(f"sample_data|{public_scene}|lidar|{timestamp_us}")
        pose_token = token(f"ego_pose|{public_scene}|lidar|{timestamp_us}")
        sample.append({
            "token": sample_token,
            "timestamp": timestamp_us,
            "scene_token": token(f"scene|{public_scene}"),
            "prev": "",
            "next": "",
        })
        sample_data.append({
            "token": data_token,
            "sample_token": sample_token,
            "ego_pose_token": pose_token,
            "calibrated_sensor_token": calibration_token,
            "timestamp": timestamp_us,
            "fileformat": "pcd",
            "is_key_frame": True,
            "height": 0,
            "width": 0,
            "filename": item["output_relative_path"],
            "prev": "",
            "next": "",
        })
        pose = item["rebased_pose"]
        ego_pose.append({
            "token": pose_token,
            "translation": pose.translation.astype(float).tolist(),
            "rotation": pose.rotation_wxyz.astype(float).tolist(),
            "timestamp": timestamp_us,
        })
        global_lidar = pose.transform @ lidar_calibration.transform
        global_box_rotation = quaternion_multiply(
            pose.rotation_wxyz, lidar_calibration.rotation_wxyz
        )
        for box_item in item["boxes"]:
            box = box_item["source_box"]
            opaque_track = identity_mapping[box.source_identity]
            instance_token = token(f"instance|{public_scene}|{opaque_track}")
            annotation_token = token(f"annotation|{sample_token}|{instance_token}")
            annotation_index = len(sample_annotation)
            sample_annotation.append({
                "token": annotation_token,
                "sample_token": sample_token,
                "instance_token": instance_token,
                "visibility_token": "",
                "attribute_tokens": [],
                "translation": transform_point(global_lidar, box.center).astype(float).tolist(),
                "size": [float(box.extents[1]), float(box.extents[0]), float(box.extents[2])],
                "rotation": global_box_rotation.astype(float).tolist(),
                "prev": "",
                "next": "",
                "num_lidar_pts": box_item["num_lidar_pts"],
                "num_radar_pts": 0,
            })
            annotation_indices_by_track[opaque_track].append(annotation_index)
            box_item["sample_token"] = sample_token
            box_item["sample_data_token"] = data_token
            box_item["instance_token"] = instance_token
            box_item["annotation_token"] = annotation_token
            box_item["global_translation"] = sample_annotation[-1]["translation"]
            box_item["global_rotation_wxyz"] = sample_annotation[-1]["rotation"]
            box_item["output_size_wlh"] = sample_annotation[-1]["size"]

    _link(sample, [list(range(len(sample)))])
    _link(sample_data, [list(range(len(sample_data)))])
    _link(sample_annotation, list(annotation_indices_by_track.values()))

    instance = []
    for source_identity, opaque_track in sorted(identity_mapping.items(), key=lambda pair: pair[1]):
        indices = annotation_indices_by_track.get(opaque_track, [])
        if not indices:
            continue
        instance.append({
            "token": token(f"instance|{public_scene}|{opaque_track}"),
            "category_token": category_token,
            "nbr_annotations": len(indices),
            "first_annotation_token": sample_annotation[indices[0]]["token"],
            "last_annotation_token": sample_annotation[indices[-1]]["token"],
        })

    scene_token = token(f"scene|{public_scene}")
    scene = [{
        "token": scene_token,
        "log_token": log_token,
        "nbr_samples": len(sample),
        "first_sample_token": sample[0]["token"] if sample else "",
        "last_sample_token": sample[-1]["token"] if sample else "",
        "name": public_scene,
        "description": f"{location.replace('_', ' ').title()} recording containing human activity.",
    }]
    splits = {"aghri_train": [], "aghri_val": [], "aghri_test": []}
    splits[f"aghri_{resolution.split}"] = [public_scene]

    return {
        "category": category,
        "attribute": attribute,
        "visibility": visibility,
        "sensor": sensor,
        "calibrated_sensor": calibrated_sensor,
        "ego_pose": ego_pose,
        "log": log,
        "scene": scene,
        "sample": sample,
        "sample_data": sample_data,
        "instance": instance,
        "sample_annotation": sample_annotation,
        "map": [],
        "splits": splits,
    }
