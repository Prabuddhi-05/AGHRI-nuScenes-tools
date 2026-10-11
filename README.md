# AGHRI-to-nuScenes-Style Conversion and PKL Generation Tools

This repository provides the tools used to convert the **AGHRI dataset into a nuScenes-style representation** and to generate legacy MMDetection3D/BEVFusion information PKLs for camera–LiDAR perception workflows.

The repository contains:

- AGHRI-to-nuScenes-style conversion code;
- fixed train, validation, and test split definitions;
- camera–LiDAR synchronization handling used by the conversion;
- PKL-generation code for three AGHRI sensor configurations; and
- documentation of the converted JSON tables, sensor files, coordinate conventions, and PKL structure.

The converted dataset and information PKLs are generated locally from the original AGHRI dataset. The conversion output is then used by [AGHRI-BEVFusion](https://github.com/Prabuddhi-05/AGHRI-BEVFusion).

## Resources

| Resource | Link |
|---|---|
| AGHRI paper preprint | [SSRN preprint](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7360013) |
| Original AGHRI dataset | [University of Lincoln Research Repository](https://doi.org/10.24385/lincoln.32982638) |
| Conversion and PKL-generation tools | [AGHRI-nuScenes-tools](https://github.com/Prabuddhi-05/AGHRI-nuScenes-tools) |
| General AGHRI dataset tools | [LCAS/AGHRI-dataset-tools](https://github.com/LCAS/AGHRI-dataset-tools) |
| BEVFusion adaptation for AGHRI | [AGHRI-BEVFusion](https://github.com/Prabuddhi-05/AGHRI-BEVFusion) |

> **Data preparation:** Download the original AGHRI dataset from the DOI above. This repository creates the converted dataset and PKLs in your chosen output directories; it does not contain sensor data or generated PKLs.

---

## What This Repository Provides

The conversion prepares AGHRI camera, LiDAR, pose, calibration, synchronization, and 3D annotation information using a relational structure based on the nuScenes data model while retaining documented AGHRI-specific conventions.

The completed conversion contains:

- **65 AGHRI scenes**: 52 train, 7 validation, and 6 test;
- **13,605 LiDAR-anchored samples**;
- **141,305 sensor `sample_data` records**;
- **26,637 3D human annotations**;
- **136 tracked human instances**;
- one ZED RGB camera;
- front, left, and right fisheye cameras; and
- one LiDAR channel.

Two AGHRI-specific metadata tables are added to the relational representation:

- `splits.json` — records the fixed AGHRI train/validation/test split;
- `camera_model.json` — records camera model and distortion information that is not represented by the standard camera intrinsic matrix alone.

The conversion also writes per-scene camera–LiDAR synchronization metadata under `metadata/synchronization/`.

---

## Documentation

Detailed documentation is provided separately for the two main stages.

| Guide | Main contents |
|---|---|
| [`docs/CONVERTED_DATA_FORMAT.md`](docs/CONVERTED_DATA_FORMAT.md) | AGHRI-to-nuScenes-style conversion, dataset structure, JSON tables, calibration, annotations, and synchronization |
| [`docs/PKL_FORMAT.md`](docs/PKL_FORMAT.md) | PKL generation, supported sensor variants, and PKL fields |

---

## Repository Layout

```text
.
├── README.md
├── requirements.txt
├── paths.env.example
├── scripts/
│   ├── discover.sh
│   ├── preview_conversion.sh
│   ├── run_conversion.sh
│   └── generate_pkls.sh
├── conversion/
│   ├── full_discovery.py
│   ├── full_conversion.py
│   ├── configs/
│   └── aghri_nuscenes/
├── splits/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
├── pkl_generation/
│   ├── pkl_generator.py
│   ├── pkl_annotations.py
│   ├── pkl_contract.py
│   ├── pkl_splits.py
│   └── pkl_transforms.py
└── docs/
    ├── CONVERTED_DATA_FORMAT.md
    └── PKL_FORMAT.md
```

`full_discovery.py` performs the required read-only preflight before conversion. `full_conversion.py` carries out the AGHRI-to-nuScenes-style conversion. The `pkl_generation/` modules generate information PKLs from an already converted AGHRI dataset.

---

## Installation

Clone the repository and create an isolated Python environment:

```bash
git clone https://github.com/Prabuddhi-05/AGHRI-nuScenes-tools.git
cd AGHRI-nuScenes-tools

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The conversion was developed and checked with Python 3.10.12.

---

## Configure Paths

Copy the path template:

```bash
cp paths.env.example paths.env
```

Edit the paths in `paths.env`:

```bash
AGHRI_SOURCE_ROOT="/absolute/path/to/extracted/AGHRI"
CONVERSION_RUN_ROOT="/absolute/path/to/new/aghri_nuscenes_run"
AGHRI_DATASET_ROOT="$CONVERSION_RUN_ROOT/AGHRI_nuScenes_camera_lidar_full"
CONVERSION_WORK_ROOT="$CONVERSION_RUN_ROOT/work"
AGHRI_PKL_OUTPUT_ROOT="/absolute/path/to/new/aghri_pkls"
```

---

# Part I — AGHRI-to-nuScenes-Style Conversion

## Prepare the Original AGHRI Dataset

Download all ten original AGHRI `dataset_part*.zip` archives and `calibration.zip` from the [official dataset record](https://doi.org/10.24385/lincoln.32982638). Extract them under one `AGHRI_SOURCE_ROOT`. The converter resolves the 65 scene directories inside `dataset_part*` directories and reads calibration from the root-level `calibration/` directory. Keep the original scene folder names and files intact.

The converter expects this layout. If an archive extracts recording folders directly, place those folders under its corresponding `dataset_partN/` directory. If it already contains the `dataset_partN/` prefix, extract it at `AGHRI_SOURCE_ROOT` to avoid an extra directory level.

```text
AGHRI_SOURCE_ROOT/
├── calibration/
│   ├── intrinsics.json
│   └── extrinsics.json
├── dataset_part1/
│   └── <recording>_label/
│       ├── sensor_data/
│       │   ├── lidar/
│       │   ├── cam_zed_rgb/
│       │   ├── cam_fish_front/
│       │   ├── cam_fish_left/
│       │   └── cam_fish_right/
│       ├── annotations/lidar_ann.json
│       └── metadata/
│           ├── odom_global.jsonl
│           └── tf/
│               ├── map__to__odom.jsonl
│               └── odom__to__base_link.jsonl
├── dataset_part2/
└── ... dataset_part10/
```

The fixed sequence-level split lists are stored under `splits/`. Discovery checks that all 65 required scenes, calibration files, and reviewed annotation inputs match the configured conversion policy. If it fails, inspect the source extraction and its reported mismatch before running conversion.

## 1. Run Discovery

The discovery stage checks the source dataset and resolves the 65 recordings before conversion:

```bash
./scripts/discover.sh
```

A successful discovery writes `$CONVERSION_RUN_ROOT/discovery/inventory.json` with:

```text
"decision": "PASS"
"resolved_scene_count": 65
```

## 2. Preview the Conversion

Run the conversion in dry-run mode:

```bash
./scripts/preview_conversion.sh
```

This checks the requested conversion without creating the converted dataset files.

## 3. Run the Conversion

```bash
./scripts/run_conversion.sh
```

The conversion:

- uses validated annotated LiDAR frames as the main sample anchors;
- converts the selected LiDAR point clouds into five-column binary files;
- copies accepted camera images without rectifying or undistorting them;
- computes sensor and ego transforms;
- builds the AGHRI metadata in a nuScenes-style relational structure;
- creates tracked human annotations;
- stores the fixed AGHRI split information; and
- writes per-scene camera–LiDAR synchronization metadata.

The dataset is written to `AGHRI_DATASET_ROOT` from `paths.env`. Its `samples/` and `sweeps/` files are created by the conversion itself. The same root contains `v1.0-aghri/` with 15 JSON tables and `metadata/synchronization/` with one association file per scene. The output also contains reports and manifests; `manifests/source_to_scene.json` maps each `aghri-scene-XXXX` identifier to its original recording name. Keep the generated `samples/`, `sweeps/`, tables, and synchronization metadata together when moving the dataset.

[`docs/scene_name_map.csv`](docs/scene_name_map.csv) is a reference list for the 65 public `aghri-scene-XXXX` identifiers and their original AGHRI recording names. The converter determines these names from the ordered `splits/` lists and matching source recording directories; it does not read the CSV. Each completed conversion writes its own authoritative mapping to `manifests/source_to_scene.json`. Use that generated manifest if the input or split lists change.

For the full directory layout, calibration and camera-model conventions, and JSON-table reference, see [`docs/CONVERTED_DATA_FORMAT.md`](docs/CONVERTED_DATA_FORMAT.md).

---

# Part II — PKL Generation

Generate the PKLs from the converted dataset produced in Part I. `scripts/generate_pkls.sh` reads `AGHRI_DATASET_ROOT` and writes the variants under `AGHRI_PKL_OUTPUT_ROOT` from `paths.env`. The output root must be new or contain no existing variant `pkls/` directories.

The dataset root supplied to the generator must directly contain:

```text
samples/
sweeps/
v1.0-aghri/
metadata/synchronization/
```

## Supported Variants

| Variant | Required AGHRI cameras | Train samples | Validation samples | Test samples | Total |
|---|---|---:|---:|---:|---:|
| `full` | ZED RGB + front, left, and right fisheye | 10,468 | 1,155 | 1,260 | 12,883 |
| `zed_lidar` | ZED RGB | 10,780 | 1,156 | 1,264 | 13,200 |
| `fisheye_lidar` | front, left, and right fisheye | 10,706 | 1,183 | 1,280 | 13,169 |

The counts differ because a LiDAR-anchored sample is retained only when every camera required by the selected variant has an accepted keyframe association. The underlying scene split remains fixed.

## Generate PKLs

Generate all three variants after conversion finishes:

```bash
./scripts/generate_pkls.sh all
```

For BEVFusion, use `AGHRI_DATASET_ROOT` as `--dataset-root` and `AGHRI_PKL_OUTPUT_ROOT` as `--pkl-root`. The two roots can be in different directories. Keep the variant folders shown below under the PKL root.

Or generate one variant:

```bash
./scripts/generate_pkls.sh full
./scripts/generate_pkls.sh zed_lidar
./scripts/generate_pkls.sh fisheye_lidar
```

Expected files include:

```text
full/pkls/aghri_full_infos_train.pkl
full/pkls/aghri_full_infos_val.pkl
full/pkls/aghri_full_infos_test.pkl

zed_lidar/pkls/aghri_zed_lidar_infos_train.pkl
zed_lidar/pkls/aghri_zed_lidar_infos_val.pkl
zed_lidar/pkls/aghri_zed_lidar_infos_test.pkl

fisheye_lidar/pkls/aghri_fisheye_lidar_infos_train.pkl
fisheye_lidar/pkls/aghri_fisheye_lidar_infos_val.pkl
fisheye_lidar/pkls/aghri_fisheye_lidar_infos_test.pkl
```

Each PKL uses the legacy information-file structure:

```python
{
    "infos": [...],
    "metadata": {...},
}
```

The generated sensor paths are relative to the AGHRI-to-nuScenes-style dataset root, so local absolute dataset paths are not embedded in the PKLs.

For the complete PKL field reference, see
[`docs/PKL_FORMAT.md`](docs/PKL_FORMAT.md).

---

## Compatibility

The output follows the nuScenes relational model where practical, but it remains an **AGHRI-to-nuScenes-style representation**.

AGHRI-specific differences include its sensor channel names, camera distortion metadata, empty map table, `camera_model.json`, and `splits.json`. These differences are documented in the conversion and PKL guides.

---

## Related Projects

- [AGHRI-dataset-tools](https://github.com/LCAS/AGHRI-dataset-tools) — general-purpose tools for the original AGHRI release.
- [AGHRI-BEVFusion](https://github.com/Prabuddhi-05/AGHRI-BEVFusion) — training and evaluation using the converted dataset and generated PKLs.
- [MMDetection3D nuScenes documentation](https://github.com/open-mmlab/mmdetection3d/blob/1.0/docs/en/datasets/nuscenes_det.md) — reference for the legacy information-file workflow.

---

## AGHRI Dataset and Citation

These tools are designed for:

**AGHRI: A dataset for multimodal robot perception of humans in agricultural and off-road environments**

- **Original dataset:** [University of Lincoln Research Repository](https://doi.org/10.24385/lincoln.32982638)
- **Submitted manuscript / preprint:** [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7360013)

When using these tools with AGHRI, please cite the original AGHRI dataset and the associated manuscript where appropriate. Use the dataset and manuscript pages for the current author list, dataset version, and preferred citation information.
