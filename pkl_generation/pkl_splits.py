"""Resolve the converted dataset splits without inventing another partition."""

from __future__ import annotations

import json
from pathlib import Path

from pkl_contract import SPLIT_KEY_TO_NAME


def load_dataset_splits(version_dir: Path) -> tuple[dict[str, str], dict]:
    raw = json.loads((version_dir / "splits.json").read_text(encoding="utf-8"))
    if set(raw) != set(SPLIT_KEY_TO_NAME):
        raise ValueError(f"unexpected split keys: {list(raw)}")
    membership: dict[str, str] = {}
    for key, split in SPLIT_KEY_TO_NAME.items():
        for scene in raw[key]:
            if scene in membership:
                raise ValueError(f"scene appears in multiple splits: {scene}")
            membership[scene] = split
    return membership, raw
