"""Strict AGHRI binary PCD decoding and nuScenes LiDAR payload encoding."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PCDHeader:
    fields: tuple[str, ...]
    sizes: tuple[int, ...]
    types: tuple[str, ...]
    counts: tuple[int, ...]
    width: int
    height: int
    points: int
    data: str
    header_bytes: int

    def serializable(self) -> dict:
        value = asdict(self)
        value["fields"] = list(self.fields)
        value["sizes"] = list(self.sizes)
        value["types"] = list(self.types)
        value["counts"] = list(self.counts)
        return value


def parse_pcd_header(path: Path) -> tuple[PCDHeader, bytes]:
    values: dict[str, list[str]] = {}
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"PCD header has no DATA line: {path}")
            try:
                decoded = line.decode("ascii").strip()
            except UnicodeDecodeError as error:
                raise ValueError(f"non-ASCII PCD header: {path}") from error
            if not decoded or decoded.startswith("#"):
                continue
            key, *parts = decoded.split()
            values[key.upper()] = parts
            if key.upper() == "DATA":
                payload = handle.read()
                header_bytes = handle.tell() - len(payload)
                break
    required = {"FIELDS", "SIZE", "TYPE", "COUNT", "WIDTH", "HEIGHT", "POINTS", "DATA"}
    missing = sorted(required - values.keys())
    if missing:
        raise ValueError(f"PCD header missing {missing}: {path}")
    header = PCDHeader(
        fields=tuple(values["FIELDS"]),
        sizes=tuple(map(int, values["SIZE"])),
        types=tuple(values["TYPE"]),
        counts=tuple(map(int, values["COUNT"])),
        width=int(values["WIDTH"][0]),
        height=int(values["HEIGHT"][0]),
        points=int(values["POINTS"][0]),
        data=values["DATA"][0].lower(),
        header_bytes=header_bytes,
    )
    return header, payload


def decode_aghri_pcd(path: Path) -> tuple[np.ndarray, PCDHeader, dict]:
    header, payload = parse_pcd_header(path)
    expected = PCDHeader(
        fields=("x", "y", "z", "rgb"), sizes=(4, 4, 4, 4),
        types=("F", "F", "F", "F"), counts=(1, 1, 1, 1),
        width=header.width, height=header.height, points=header.points,
        data="binary", header_bytes=header.header_bytes,
    )
    if (header.fields, header.sizes, header.types, header.counts, header.data) != (
        expected.fields, expected.sizes, expected.types, expected.counts, expected.data
    ):
        raise ValueError(f"unsupported PCD layout in {path}: {header.serializable()}")
    if header.width * header.height != header.points:
        raise ValueError(f"inconsistent PCD dimensions in {path}")
    expected_bytes = header.points * 4 * 4
    if len(payload) != expected_bytes:
        raise ValueError(
            f"PCD payload length mismatch in {path}: {len(payload)} != {expected_bytes}"
        )
    raw = np.frombuffer(payload, dtype="<f4").reshape(header.points, 4)
    xyz = raw[:, :3]
    finite = np.all(np.isfinite(xyz), axis=1)
    exact_zero = np.all(xyz == 0.0, axis=1)
    keep = finite & ~exact_zero
    filtered = np.asarray(xyz[keep], dtype=np.float32)
    stats = {
        "declared_points": header.points,
        "decoded_points": int(raw.shape[0]),
        "nonfinite_xyz_removed": int(np.count_nonzero(~finite)),
        "exact_zero_xyz_removed": int(np.count_nonzero(finite & exact_zero)),
        "output_points": int(filtered.shape[0]),
        "xyz_min": filtered.min(axis=0).astype(float).tolist() if len(filtered) else None,
        "xyz_max": filtered.max(axis=0).astype(float).tolist() if len(filtered) else None,
    }
    return filtered, header, stats


def encode_nuscenes_lidar(points_xyz: np.ndarray) -> bytes:
    points = np.asarray(points_xyz)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("XYZ point array must be N x 3")
    if not np.all(np.isfinite(points)):
        raise ValueError("output XYZ contains non-finite values")
    output = np.zeros((points.shape[0], 5), dtype="<f4")
    output[:, :3] = points.astype("<f4", copy=False)
    return output.tobytes(order="C")


def decode_nuscenes_lidar(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    if len(payload) % 20:
        raise ValueError(f"converted payload is not N x 5 float32: {path}")
    return np.frombuffer(payload, dtype="<f4").reshape(-1, 5)
