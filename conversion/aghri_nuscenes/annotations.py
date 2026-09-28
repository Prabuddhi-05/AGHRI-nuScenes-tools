"""LiDAR annotation source resolution and strict box validation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceBox:
    record_index: int
    label_index: int
    source_identity: str
    center: tuple[float, float, float]
    extents: tuple[float, float, float]
    raw_rotation: tuple[float, float, float]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_annotation_source(
    scene_path: Path, scene_name: str, override: dict[str, Any] | None
) -> tuple[Path, str, list[str]]:
    candidates = sorted(scene_path.joinpath("annotations").glob("lidar_ann*.json"))
    candidate_names = [p.relative_to(scene_path).as_posix() for p in candidates]
    if override:
        selected = scene_path / override["selected"]
        if selected not in candidates or not selected.is_file():
            raise ValueError(f"reviewed annotation override does not resolve: {selected}")
        actual_hash = sha256_file(selected)
        expected_hash = override.get("expected_sha256")
        if expected_hash and actual_hash != expected_hash:
            raise ValueError(
                f"annotation override hash mismatch: {actual_hash} != {expected_hash}"
            )
        return selected, str(override["reason"]), candidate_names
    if len(candidates) != 1:
        raise ValueError(
            f"ambiguous_annotation_source for {scene_name}: {candidate_names}"
        )
    return candidates[0], "Only LiDAR annotation candidate present.", candidate_names


def load_annotation_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError("LiDAR annotation file must contain a JSON list")
    return value


def validate_box(
    record_index: int, label_index: int, label: Any
) -> tuple[SourceBox | None, dict[str, Any] | None]:
    base = {"record_index": record_index, "label_index": label_index}
    if not isinstance(label, dict):
        return None, {**base, "reason": "label_not_object", "observed": repr(label)[:500]}
    source_identity = label.get("Class")
    values = label.get("BoundingBoxes")
    if not isinstance(source_identity, str) or not source_identity:
        return None, {**base, "reason": "missing_or_invalid_class", "observed": source_identity}
    if not isinstance(values, list) or len(values) != 9:
        return None, {**base, "reason": "box_not_flat_nine_value_list", "observed": values}
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
        return None, {**base, "reason": "box_contains_non_numeric_value", "observed": values}
    numeric = tuple(float(v) for v in values)
    if not all(math.isfinite(v) for v in numeric):
        return None, {**base, "reason": "box_contains_nonfinite_value", "observed": values}
    if not all(v > 0.0 for v in numeric[3:6]):
        return None, {**base, "reason": "box_has_nonpositive_extent", "observed": values}
    return SourceBox(
        record_index=record_index,
        label_index=label_index,
        source_identity=source_identity,
        center=numeric[0:3],
        extents=numeric[3:6],
        raw_rotation=numeric[6:9],
    ), None


def count_points_in_box(points_xyz, box: SourceBox, tolerance: float) -> int:
    import numpy as np

    points = np.asarray(points_xyz)
    center = np.asarray(box.center)
    half_extent = np.asarray(box.extents) / 2.0 + tolerance
    return int(np.count_nonzero(np.all(np.abs(points - center) <= half_extent, axis=1)))
