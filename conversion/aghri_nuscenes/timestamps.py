"""Exact filename timestamp parsing."""

from __future__ import annotations

import re
from pathlib import Path


PCD_TIMESTAMP_RE = re.compile(r"^(?P<seconds>\d+)_(?P<nanoseconds>\d{9})\.pcd$")
OUTPUT_TIMESTAMP_RE = re.compile(
    r"^[^/]+__(?P<seconds>\d+)_(?P<nanoseconds>\d{9})\.pcd\.bin$"
)


def parse_pcd_filename(filename: str) -> tuple[int, int, int]:
    """Return seconds, nanoseconds and exact epoch nanoseconds."""
    name = Path(filename).name
    match = PCD_TIMESTAMP_RE.fullmatch(name)
    if match is None:
        raise ValueError(f"invalid PCD timestamp filename: {filename!r}")
    seconds = int(match.group("seconds"))
    nanoseconds = int(match.group("nanoseconds"))
    if not 0 <= nanoseconds < 1_000_000_000:
        raise ValueError(f"nanoseconds out of range: {filename!r}")
    return seconds, nanoseconds, seconds * 1_000_000_000 + nanoseconds


def nanoseconds_to_microseconds(timestamp_ns: int) -> tuple[int, int]:
    """Floor-divide epoch nanoseconds and return the discarded remainder."""
    return timestamp_ns // 1_000, timestamp_ns % 1_000


def parse_output_filename(filename: str) -> tuple[int, int, int]:
    match = OUTPUT_TIMESTAMP_RE.fullmatch(Path(filename).name)
    if match is None:
        raise ValueError(f"invalid converted LiDAR filename: {filename!r}")
    seconds = int(match.group("seconds"))
    nanoseconds = int(match.group("nanoseconds"))
    return seconds, nanoseconds, seconds * 1_000_000_000 + nanoseconds
