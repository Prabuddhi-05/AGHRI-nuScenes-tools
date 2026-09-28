"""Safe LiDAR-anchored synchronization for AGHRI camera streams."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Iterable

import numpy as np


SENSOR_TIMESTAMP_RE = re.compile(
    r"^(?P<seconds>\d+)_(?P<nanoseconds>\d{9})(?P<extension>\.[^.]+)$"
)


@dataclass(frozen=True)
class TimedFile:
    path: Path
    name: str
    timestamp_ns: int
    extension: str


def parse_sensor_filename(filename: str) -> tuple[int, int, int, str]:
    """Parse ``seconds_nanoseconds.ext`` without floating-point arithmetic."""
    match = SENSOR_TIMESTAMP_RE.fullmatch(Path(filename).name)
    if match is None:
        raise ValueError(f"invalid sensor timestamp filename: {filename!r}")
    seconds = int(match.group("seconds"))
    nanoseconds = int(match.group("nanoseconds"))
    if not 0 <= nanoseconds < 1_000_000_000:
        raise ValueError(f"nanoseconds out of range: {filename!r}")
    extension = match.group("extension").lower()
    return seconds, nanoseconds, seconds * 1_000_000_000 + nanoseconds, extension


def scan_timed_files(directory: Path, extensions: Iterable[str]) -> tuple[list[TimedFile], list[str]]:
    """Return timestamped files and names rejected solely by filename syntax."""
    allowed = {extension.lower() for extension in extensions}
    timed: list[TimedFile] = []
    unparseable: list[str] = []
    if not directory.is_dir():
        return [], []
    for path in sorted(item for item in directory.iterdir() if item.is_file()):
        if path.suffix.lower() not in allowed:
            continue
        try:
            _, _, timestamp_ns, extension = parse_sensor_filename(path.name)
        except ValueError:
            unparseable.append(path.name)
            continue
        timed.append(TimedFile(path, path.name, timestamp_ns, extension))
    timed.sort(key=lambda item: (item.timestamp_ns, item.name))
    return timed, unparseable


def nearest_index(sorted_timestamps: np.ndarray, timestamp_ns: int) -> int:
    if sorted_timestamps.ndim != 1 or sorted_timestamps.size == 0:
        raise ValueError("nearest_index requires a nonempty one-dimensional array")
    index = int(np.searchsorted(sorted_timestamps, timestamp_ns))
    if index == 0:
        return 0
    if index == sorted_timestamps.size:
        return int(sorted_timestamps.size - 1)
    before = index - 1
    return index if abs(int(sorted_timestamps[index]) - timestamp_ns) < abs(
        int(sorted_timestamps[before]) - timestamp_ns
    ) else before


def estimate_constant_offset_ns(
    lidar_timestamps_ns: np.ndarray, camera_timestamps_ns: np.ndarray
) -> tuple[int, list[int]]:
    """Use the reference synchronizer's nearest-frame median-offset estimator."""
    if lidar_timestamps_ns.ndim != 1 or camera_timestamps_ns.ndim != 1:
        raise ValueError("timestamp inputs must be one-dimensional")
    if lidar_timestamps_ns.size == 0 or camera_timestamps_ns.size == 0:
        raise ValueError("offset estimation requires nonempty streams")
    differences = [
        int(camera_timestamps_ns[nearest_index(camera_timestamps_ns, int(timestamp))])
        - int(timestamp)
        for timestamp in lidar_timestamps_ns
    ]
    return int(np.median(np.asarray(differences, dtype=np.int64))), differences


def monotonic_one_to_one_match(
    anchor_timestamps_ns: np.ndarray,
    target_timestamps_ns: np.ndarray,
    threshold_ns: int,
) -> tuple[list[int], list[float]]:
    """Greedily match nearest strictly increasing target indices.

    The supplied reference matcher can select ``j-1`` after selecting ``j`` and
    therefore reports backward order violations even with monotonic mode on.
    Conversion requires both one-to-one use and zero order violations, so the lower
    target bound here is always the index after the last accepted target.
    """
    if threshold_ns < 0:
        raise ValueError("threshold must be nonnegative")
    if anchor_timestamps_ns.ndim != 1 or target_timestamps_ns.ndim != 1:
        raise ValueError("timestamp inputs must be one-dimensional")
    if any(b <= a for a, b in zip(anchor_timestamps_ns, anchor_timestamps_ns[1:])):
        raise ValueError("anchor timestamps must be strictly increasing")
    if any(b <= a for a, b in zip(target_timestamps_ns, target_timestamps_ns[1:])):
        raise ValueError("target timestamps must be strictly increasing")

    matched: list[int] = []
    residual_s: list[float] = []
    last_accepted = -1
    for anchor in anchor_timestamps_ns:
        lower = last_accepted + 1
        if lower >= target_timestamps_ns.size:
            matched.append(-1)
            residual_s.append(math.nan)
            continue
        position = int(np.searchsorted(target_timestamps_ns, int(anchor)))
        candidates = {
            max(lower, position - 1),
            max(lower, position),
        }
        candidates = {
            index for index in candidates if lower <= index < target_timestamps_ns.size
        }
        if not candidates:
            matched.append(-1)
            residual_s.append(math.nan)
            continue
        selected = min(
            candidates,
            key=lambda index: (
                abs(int(target_timestamps_ns[index]) - int(anchor)), index
            ),
        )
        residual_ns = abs(int(target_timestamps_ns[selected]) - int(anchor))
        if residual_ns <= threshold_ns:
            matched.append(selected)
            residual_s.append(residual_ns / 1_000_000_000.0)
            last_accepted = selected
        else:
            matched.append(-1)
            residual_s.append(math.nan)
    return matched, residual_s


def _distribution(values: list[float]) -> dict[str, float | None]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {"mean_s": None, "p95_s": None, "p99_s": None, "max_s": None}
    return {
        "mean_s": float(np.mean(finite)),
        "p95_s": float(np.percentile(finite, 95)),
        "p99_s": float(np.percentile(finite, 99)),
        "max_s": float(np.max(finite)),
    }


def evaluate_threshold(
    lidar_timestamps_ns: np.ndarray,
    aligned_camera_timestamps: dict[str, np.ndarray],
    threshold_s: float,
) -> tuple[dict, dict[str, tuple[list[int], list[float]]]]:
    threshold_ns = int(round(threshold_s * 1_000_000_000))
    matches: dict[str, tuple[list[int], list[float]]] = {}
    per_camera: dict[str, dict] = {}
    all_residuals: list[float] = []
    mutual_hits = 0
    mutual_total = 0
    order_violations = 0
    for channel, timestamps in aligned_camera_timestamps.items():
        indices, residuals = monotonic_one_to_one_match(
            lidar_timestamps_ns, timestamps, threshold_ns
        )
        matches[channel] = (indices, residuals)
        accepted_indices = [index for index in indices if index >= 0]
        for lidar_index, camera_index in enumerate(indices):
            if camera_index < 0:
                continue
            mutual_total += 1
            if nearest_index(lidar_timestamps_ns, int(timestamps[camera_index])) == lidar_index:
                mutual_hits += 1
        order_violations += sum(
            right <= left for left, right in zip(accepted_indices, accepted_indices[1:])
        )
        finite = [value for value in residuals if math.isfinite(value)]
        all_residuals.extend(finite)
        per_camera[channel] = {
            "accepted": len(accepted_indices),
            "missing": int(lidar_timestamps_ns.size - len(accepted_indices)),
            "percent_matched": len(accepted_indices) / lidar_timestamps_ns.size * 100.0,
            "unique_selected_images": len(set(accepted_indices)),
            "order_violations": sum(
                right <= left for left, right in zip(accepted_indices, accepted_indices[1:])
            ),
            **_distribution(finite),
        }
    complete = sum(
        all(matches[channel][0][index] >= 0 for channel in matches)
        for index in range(lidar_timestamps_ns.size)
    )
    distribution = _distribution(all_residuals)
    metrics = {
        "threshold_s": float(threshold_s),
        "complete_samples": complete,
        "percent_complete": complete / lidar_timestamps_ns.size * 100.0,
        "average_absolute_residual_s": distribution["mean_s"],
        "p95_residual_s": distribution["p95_s"],
        "p99_residual_s": distribution["p99_s"],
        "maximum_residual_s": distribution["max_s"],
        "mutual_nearest_neighbour_rate": (
            mutual_hits / mutual_total if mutual_total else 0.0
        ),
        "order_violations": order_violations,
        "per_camera": per_camera,
    }
    return metrics, matches


def choose_threshold(
    candidates: list[dict], p95_limit_s: float, plateau_eps: float
) -> tuple[dict, bool]:
    """Apply the supplied synchronizer's p95/plateau/MNN/order/threshold ranking."""
    if not candidates:
        raise ValueError("no threshold candidates")

    def pick(values: list[dict]) -> dict:
        maximum_completeness = max(item["percent_complete"] for item in values)
        plateau = [
            item for item in values
            if maximum_completeness - item["percent_complete"] <= plateau_eps
        ]
        return sorted(
            plateau,
            key=lambda item: (
                -item["mutual_nearest_neighbour_rate"],
                item["order_violations"],
                item["threshold_s"],
            ),
        )[0]

    preferred = [
        item for item in candidates
        if item["p95_residual_s"] is not None
        and item["p95_residual_s"] <= p95_limit_s
    ]
    return (pick(preferred), True) if preferred else (pick(candidates), False)


def synchronize(
    lidar_files: list[TimedFile],
    camera_files: dict[str, list[TimedFile]],
    thresholds_s: list[float],
    p95_limit_s: float,
    plateau_eps: float,
    offset_camera_files: dict[str, list[TimedFile]] | None = None,
) -> dict:
    """Estimate offsets, evaluate thresholds and return detailed final matches."""
    lidar_ns = np.asarray([item.timestamp_ns for item in lidar_files], dtype=np.int64)
    offsets: dict[str, int] = {}
    estimator_differences: dict[str, list[int]] = {}
    aligned: dict[str, np.ndarray] = {}
    for channel, files in camera_files.items():
        estimator_files = (
            offset_camera_files[channel] if offset_camera_files is not None else files
        )
        estimator_original = np.asarray(
            [item.timestamp_ns for item in estimator_files], dtype=np.int64
        )
        original = np.asarray([item.timestamp_ns for item in files], dtype=np.int64)
        offset, differences = estimate_constant_offset_ns(lidar_ns, estimator_original)
        offsets[channel] = offset
        estimator_differences[channel] = differences
        aligned[channel] = original - offset

    candidate_metrics = [
        evaluate_threshold(lidar_ns, aligned, threshold)[0]
        for threshold in thresholds_s
    ]
    selected_metrics, p95_preference_met = choose_threshold(
        candidate_metrics, p95_limit_s, plateau_eps
    )
    final_metrics, final_matches = evaluate_threshold(
        lidar_ns, aligned, selected_metrics["threshold_s"]
    )

    rows = []
    for lidar_index, lidar_file in enumerate(lidar_files):
        cameras = {}
        for channel, files in camera_files.items():
            camera_index = final_matches[channel][0][lidar_index]
            if camera_index < 0:
                cameras[channel] = {
                    "selected_filename": None,
                    "original_timestamp_ns": None,
                    "aligned_timestamp_ns": None,
                    "absolute_aligned_residual_ns": None,
                    "rejection_reason": "no_unused_monotonic_image_within_selected_threshold",
                }
                continue
            camera = files[camera_index]
            aligned_ns = camera.timestamp_ns - offsets[channel]
            cameras[channel] = {
                "selected_filename": camera.name,
                "original_timestamp_ns": camera.timestamp_ns,
                "aligned_timestamp_ns": aligned_ns,
                "absolute_aligned_residual_ns": abs(aligned_ns - lidar_file.timestamp_ns),
                "rejection_reason": None,
            }
        rows.append({
            "lidar_index": lidar_index,
            "lidar_filename": lidar_file.name,
            "lidar_timestamp_ns": lidar_file.timestamp_ns,
            "cameras": cameras,
        })

    return {
        "algorithm": {
            "anchor": "lidar",
            "offset_estimator": "median(nearest_original_camera_timestamp - lidar_timestamp)",
            "offset_estimator_allows_reuse": True,
            "offset_estimator_population": (
                "all source camera timestamps, before per-measurement pose exclusions"
                if offset_camera_files is not None else "all supplied camera timestamps"
            ),
            "final_matcher": "nearest unused strictly chronological aligned camera frame",
            "one_to_one": True,
            "enforce_monotonic": True,
            "p95_limit_s": p95_limit_s,
            "plateau_eps_percentage_points": plateau_eps,
        },
        "estimated_camera_offsets_ns": offsets,
        "offset_estimator_differences_ns": estimator_differences,
        "threshold_candidates": candidate_metrics,
        "chosen_threshold_s": selected_metrics["threshold_s"],
        "p95_preference_met": p95_preference_met,
        "chosen_threshold_metrics": final_metrics,
        "samples": rows,
    }
