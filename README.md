# AGHRI-to-nuScenes-Style Conversion and PKL Generation Tools

This repository provides the tools used to convert the **AGHRI dataset into a nuScenes-style representation** and to generate legacy MMDetection3D/BEVFusion information PKLs for camera–LiDAR perception workflows.

The repository contains:

- AGHRI-to-nuScenes-style conversion code;
- fixed train, validation, and test split definitions;
- camera–LiDAR synchronization handling used by the conversion;
- PKL-generation code for three AGHRI sensor configurations; and
- documentation of the converted JSON tables, sensor files, coordinate conventions, and PKL structure.

The converted dataset and pre-generated PKLs are maintained separately in the **AGHRI-nuScenes-release** repository.

## Resources

| Resource | Link |
|---|---|
| AGHRI paper preprint | [SSRN preprint](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7360013) |
| Original AGHRI dataset | [University of Lincoln Research Repository](https://doi.org/10.24385/lincoln.32982638) |
| AGHRI-to-nuScenes-style release | [AGHRI-nuScenes-release](https://github.com/Prabuddhi-05/AGHRI-nuScenes-release) |
| Conversion and PKL-generation tools | [AGHRI-nuScenes-tools](https://github.com/Prabuddhi-05/AGHRI-nuScenes-tools) |
| General AGHRI dataset tools | [LCAS/AGHRI-dataset-tools](https://github.com/LCAS/AGHRI-dataset-tools) |
| BEVFusion adaptation for AGHRI | [AGHRI-BEVFusion](https://github.com/Prabuddhi-05/AGHRI-BEVFusion) |

> **Important:** The original AGHRI dataset and the converted sensor payloads are not stored in this tools repository. Download the original AGHRI data from the official dataset record to run a new conversion, or download the converted AGHRI-to-nuScenes-style release if you only need the prepared dataset and PKLs.

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

Download AGHRI from the [official dataset record](https://doi.org/10.24385/lincoln.32982638) and extract the required dataset archives and calibration files into one dataset root.

The conversion expects the AGHRI release structure, including:

```text
AGHRI_SOURCE_ROOT/
├── calibration/
│   ├── intrinsics.json
│   └── extrinsics.json
└── <recording>_label/
    ├── sensor_data/
    │   ├── lidar/
    │   ├── cam_zed_rgb/
    │   ├── cam_fish_front/
    │   ├── cam_fish_left/
    │   └── cam_fish_right/
    ├── annotations/
    │   └── lidar_ann.json
    └── metadata/
        ├── odom_global.jsonl
        └── tf/
            ├── map__to__odom.jsonl
            └── odom__to__base_link.jsonl
```

The fixed sequence-level split lists are stored under `splits/`.

## 1. Run Discovery

The discovery stage checks the source dataset and resolves the 65 recordings before conversion:

```bash
./scripts/discover.sh
```

A successful discovery should report:

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

For the full conversion and JSON-table reference, see
[`docs/CONVERTED_DATA_FORMAT.md`](docs/CONVERTED_DATA_FORMAT.md).

---

# Part II — PKL Generation

Pre-generated PKLs for the released dataset can be downloaded from the [AGHRI-nuScenes-release](https://github.com/Prabuddhi-05/AGHRI-nuScenes-release) repository.

This tools repository can also be used to **regenerate the PKLs** from an AGHRI-to-nuScenes-style converted dataset.

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

Generate all three variants:

```bash
./scripts/generate_pkls.sh all
```

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
- [AGHRI-nuScenes-release](https://github.com/Prabuddhi-05/AGHRI-nuScenes-release) — converted AGHRI metadata and pre-generated PKLs.
- [MMDetection3D nuScenes documentation](https://github.com/open-mmlab/mmdetection3d/blob/1.0/docs/en/datasets/nuscenes_det.md) — reference for the legacy information-file workflow.

---

## AGHRI Dataset and Citation

These tools are designed for:

**AGHRI: A dataset for multimodal robot perception of humans in agricultural and off-road environments**

- **Original dataset:** [University of Lincoln Research Repository](https://doi.org/10.24385/lincoln.32982638)
- **Submitted manuscript / preprint:** [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7360013)

When using these tools with AGHRI, please cite the original AGHRI dataset and the associated manuscript where appropriate. Use the dataset and manuscript pages for the current author list, dataset version, and preferred citation information.
