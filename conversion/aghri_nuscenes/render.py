"""Privacy-safe top-down LiDAR and box validation renders."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _render_one(item: dict[str, Any], output_path: Path, criterion: str, public_scene: str) -> None:
    points = item["points_xyz"]
    fig, axis = plt.subplots(figsize=(10, 10), dpi=140)
    axis.scatter(points[:, 0], points[:, 1], s=0.25, c="#555555", alpha=0.45, rasterized=True)
    colours = plt.cm.tab10(np.linspace(0, 1, max(1, len(item["boxes"]))))
    for colour, box_item in zip(colours, item["boxes"]):
        box = box_item["source_box"]
        cx, cy, _ = box.center
        ex, ey, _ = box.extents
        corners = np.array([
            [cx - ex / 2, cy - ey / 2], [cx + ex / 2, cy - ey / 2],
            [cx + ex / 2, cy + ey / 2], [cx - ex / 2, cy + ey / 2],
            [cx - ex / 2, cy - ey / 2],
        ])
        axis.plot(corners[:, 0], corners[:, 1], color=colour, linewidth=1.6)
        axis.scatter([cx], [cy], marker="x", s=35, color=colour)
        axis.arrow(cx, cy, ex * 0.35, 0, color=colour, width=0.008, head_width=0.08, length_includes_head=True)
        axis.arrow(cx, cy, 0, ey * 0.35, color=colour, width=0.008, head_width=0.08, length_includes_head=True)
        axis.text(cx, cy, f" {box_item['opaque_track']}\n n={box_item['num_lidar_pts']}", fontsize=7, color=colour)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("front_lidar_link x (m, forward)")
    axis.set_ylabel("front_lidar_link y (m, left)")
    axis.grid(True, linewidth=0.25, alpha=0.3)
    axis.set_title(
        f"{public_scene} | {criterion}\n"
        f"timestamp {item['timestamp_us']} µs | source record {item['record_index']}"
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def render_visual_checks(processed: list[dict], report_dir: Path, public_scene: str) -> list[dict]:
    visual_dir = report_dir / "visual_checks"
    visual_dir.mkdir(parents=True, exist_ok=True)
    selections: list[tuple[str, dict]] = [
        ("first", processed[0]),
        ("middle", processed[len(processed) // 2]),
        ("last", processed[-1]),
    ]
    all_boxes = [(box_item["num_lidar_pts"], item) for item in processed for box_item in item["boxes"]]
    if all_boxes:
        selections.append(("fewest_points_in_valid_box", min(all_boxes, key=lambda value: value[0])[1]))
        selections.append(("most_points_in_valid_box", max(all_boxes, key=lambda value: value[0])[1]))
    nonzero = [
        item for item in processed
        if any(any(value != 0.0 for value in box_item["source_box"].raw_rotation) for box_item in item["boxes"])
    ]
    if nonzero:
        selections.append(("ignored_nonzero_source_rotation", nonzero[0]))
    outputs = []
    for sequence, (criterion, item) in enumerate(selections, start=1):
        filename = f"{sequence:02d}_{criterion}__record_{item['record_index']:04d}.png"
        _render_one(item, visual_dir / filename, criterion, public_scene)
        outputs.append({
            "criterion": criterion,
            "record_index": item["record_index"],
            "filename": f"reports/visual_checks/{filename}",
        })
    if not nonzero:
        outputs.append({
            "criterion": "ignored_nonzero_source_rotation",
            "status": "not_applicable",
            "reason": "The selected pilot scene contains no nonzero raw rotation values.",
        })
    return outputs
