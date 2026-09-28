# AGHRI PKL Generation and Format

This document describes the information PKLs generated from the **AGHRI-to-nuScenes-style dataset representation**.

The PKLs are machine-readable catalogue files used to prepare AGHRI camera, LiDAR, calibration, pose, and annotation information for legacy MMDetection3D/BEVFusion-style dataset loaders.

They do **not** contain the camera images or LiDAR point clouds themselves. They contain the paths and metadata needed to locate and interpret those files.

---

## 1. PKL Generation Overview

The generation workflow is:

```text
AGHRI-to-nuScenes-style dataset
        │
        ├── v1.0-aghri/*.json
        ├── samples/
        ├── sweeps/
        └── metadata/synchronization/
        │
        ▼
AGHRI PKL generator
        │
        ├── reads split membership
        ├── selects eligible LiDAR anchors
        ├── reads required camera associations
        ├── builds transforms and camera metadata
        ├── converts 3D annotations to the key-LiDAR frame
        └── prepares velocity / point-count information
        │
        ▼
train / validation / test information PKLs
```

The generator reuses the camera–LiDAR associations already stored by the conversion. It does not independently rematch camera timestamps.

---

## 2. Relevant Modules

| Module | Purpose |
|---|---|
| `pkl_generator.py` | Main generator. Reads the converted dataset, selects eligible samples, creates information records, and writes train/validation/test PKLs |
| `pkl_contract.py` | Stores the main PKL contract, sensor order, detection class, point dimensions, and output naming |
| `pkl_splits.py` | Reads the AGHRI split information and resolves scene membership |
| `pkl_annotations.py` | Converts human boxes into the key-LiDAR frame and prepares ground-truth arrays, velocities, and point counts |
| `pkl_transforms.py` | Provides quaternion, rotation, and coordinate-transform calculations used by camera, LiDAR, pose, and annotation fields |

---

## 3. Supported Sensor Variants

| Variant | Required AGHRI cameras in loader order | Train | Validation | Test | Total |
|---|---|---:|---:|---:|---:|
| `full` | ZED RGB, front fisheye, left fisheye, right fisheye | 10,468 | 1,155 | 1,260 | 12,883 |
| `zed_lidar` | ZED RGB | 10,780 | 1,156 | 1,264 | 13,200 |
| `fisheye_lidar` | front fisheye, left fisheye, right fisheye | 10,706 | 1,183 | 1,280 | 13,169 |

A LiDAR-anchored sample is retained only when every camera required by the selected variant has an accepted keyframe association.

The scene split itself does not change between variants.

`aghri-scene-0027` has no accepted fisheye matches under the fixed conversion policy. It remains in the training split but contributes no information records to the `full` or `fisheye_lidar` variants.

---

## 4. Generate the PKLs

The simplest command is:

```bash
./scripts/generate_pkls.sh all
```

To generate one variant:

```bash
./scripts/generate_pkls.sh full
./scripts/generate_pkls.sh zed_lidar
./scripts/generate_pkls.sh fisheye_lidar
```

Equivalent direct usage follows the pattern:

```bash
python3 pkl_generation/pkl_generator.py \
  --dataset-root /path/to/AGHRI_nuScenes_camera_lidar_full \
  --output-dir /path/to/output/full/pkls \
  --report-dir /path/to/output/full/reports \
  --manifest-dir /path/to/output/full/manifests \
  --variant full
```

The dataset root must directly contain:

```text
samples/
sweeps/
v1.0-aghri/
metadata/synchronization/
```

The output PKL directory is protected against accidental replacement of an existing generation.

---

## 5. Expected Filenames

```text
full/pkls/
├── aghri_full_infos_train.pkl
├── aghri_full_infos_val.pkl
└── aghri_full_infos_test.pkl

zed_lidar/pkls/
├── aghri_zed_lidar_infos_train.pkl
├── aghri_zed_lidar_infos_val.pkl
└── aghri_zed_lidar_infos_test.pkl

fisheye_lidar/pkls/
├── aghri_fisheye_lidar_infos_train.pkl
├── aghri_fisheye_lidar_infos_val.pkl
└── aghri_fisheye_lidar_infos_test.pkl
```

The files use Pickle protocol 4 and the legacy MMDetection3D/BEVFusion information-file structure rather than the newer `data_list` / `metainfo` schema.

---

## 6. Top-Level PKL Structure

Every PKL contains:

```python
{
    "infos": [...],
    "metadata": {...},
}
```

`metadata` describes dataset-wide assumptions.

`infos` is a list in which each entry represents one eligible LiDAR-anchored sample.

---

## 7. `metadata` Reference

The metadata records the dataset-level contract required to interpret the information records.

It includes information covering:

- source dataset identity and version;
- selected sensor variant;
- ordered camera channels;
- LiDAR channel;
- detection class;
- point-cloud dimensions and semantics;
- raw-image policy;
- box and yaw convention;
- velocity policy;
- map policy;
- database-sampler policy; and
- selected keyframe-association policy.

The released PKLs use portable descriptive metadata such as:

```text
source_dataset
source_dataset_version
```

No local absolute dataset root is embedded.

The sole detection class is:

```text
human
```

The LiDAR point contract uses five values per point:

```text
[x, y, z, 0, 0]
```

The final two values are placeholders rather than measured intensity, ring, or time features.

---

## 8. `infos` Record Reference

Each `info` entry represents one LiDAR-anchored AGHRI sample.

The generated record includes the following groups of information.

### 8.1 Sample identity and LiDAR

| Field | Meaning |
|---|---|
| `lidar_path` | path to the LiDAR binary file, relative to the converted dataset root |
| `token` | sample token |
| `timestamp` | LiDAR anchor timestamp |
| `sweeps` | previous LiDAR sweeps; empty in this release |
| `num_features` | number of values stored per LiDAR point |
| `scene_name` | public AGHRI scene name |
| `source_scene_token` | source scene token used by the converted relational structure |

The LiDAR path is portable, for example:

```text
samples/lidar/aghri-scene-0001__...pcd.bin
```

No absolute machine path is required.

---

### 8.2 LiDAR and Ego Transforms

Information records expose the transforms required to relate the key LiDAR frame to the AGHRI robot ego frame and the sequence-local global frame.

Common fields include:

```text
lidar2ego_translation
lidar2ego_rotation
ego2global_translation
ego2global_rotation
```

Quaternions follow:

```text
[w, x, y, z]
```

---

## 9. Camera Entry Reference

The `cams` mapping contains exactly the cameras required by the selected variant and preserves the configured loader order.

For example, the `full` variant contains:

```text
cam_zed_rgb
cam_fish_front
cam_fish_left
cam_fish_right
```

Each camera entry contains information such as:

| Field / information | Meaning |
|---|---|
| `data_path` | path to the image relative to the converted dataset root |
| `sample_data_token` | camera sample-data token |
| `timestamp` | original camera capture timestamp |
| `sensor2ego_translation` | camera mounting translation relative to AGHRI `base_link` |
| `sensor2ego_rotation` | camera mounting rotation |
| `ego2global_translation` | robot position at the camera timestamp |
| `ego2global_rotation` | robot orientation at the camera timestamp |
| `sensor2lidar_rotation` | camera-to-key-LiDAR rotation used by the legacy loader |
| `sensor2lidar_translation` | camera-to-key-LiDAR translation |
| `cam_intrinsic` | camera intrinsic matrix |
| `camera_model` | camera model description |
| `distortion_model` | source lens-distortion model |
| `distortion_coefficients` | stored distortion coefficients |
| `distortion_coefficient_order` | coefficient interpretation/order |
| `image_state` | raw-image state |
| `image_width`, `image_height` | raw image dimensions |
| synchronization fields | stored offset, aligned timing evidence, residual, and accepted threshold |

### ZED RGB

The ZED entry contains the intrinsic matrix together with the source `rational_polynomial` distortion description.

### Fisheye Cameras

The front, left, and right fisheye entries retain the `equidistant` distortion model and fisheye distortion coefficients.

The image files themselves remain raw.

---

## 10. Ground-Truth Reference

### `gt_boxes`

Shape:

```text
N × 7
```

Format in the key-LiDAR frame:

```text
[x, y, z, x_size, y_size, z_size, yaw]
```

### `gt_names`

Contains:

```text
human
```

for every box.

### `num_lidar_pts`

Number of LiDAR points inside each ground-truth box.

### `num_radar_pts`

Always:

```text
0
```

because radar is not part of the AGHRI-to-nuScenes-style release.

### `valid_flag`

True when the corresponding LiDAR-point count is positive.

---

## 11. Ground-Truth Velocity

`gt_velocity` has shape:

```text
N × 2
```

and stores horizontal velocity in the key-LiDAR frame.

Velocity is derived from the motion of the same tracked instance between permitted neighbouring observations.

Conceptually, if:

```text
previous centre = [x_prev, y_prev] at t_prev
next centre     = [x_next, y_next] at t_next
```

then:

```text
vx = (x_next - x_prev) / (t_next - t_prev)
vy = (y_next - y_prev) / (t_next - t_prev)
```

and:

```text
gt_velocity = [vx, vy]
```

When an allowed neighbouring observation is not available, the velocity contains an explicit NaN rather than an estimated replacement value.

---

## 12. Synchronization Evidence

Each information record includes AGHRI camera–LiDAR synchronization evidence derived from the conversion.

The generator reads the already selected camera associations from:

```text
metadata/synchronization/
```

It does not rematch timestamps.

Per-camera entries can include:

```text
time_offset_s
sync_residual_s
chosen_threshold_s
```

The sample-level synchronization summary records whether the required camera set is complete and can include aggregate residual information.

This allows downstream users to inspect the timing quality associated with a selected multimodal sample.

---

## 13. Portability and Dataset Paths

The PKLs store sensor paths relative to the AGHRI-to-nuScenes-style dataset root.

Therefore a downloaded or moved dataset does **not** require PKL regeneration solely because its absolute filesystem location changes.

For example, a downstream loader can point its root to:

```text
/path/to/AGHRI-nuScenes-release/dataset
```

provided that the internal relative structure is preserved:

```text
dataset/
├── samples/
├── sweeps/
├── v1.0-aghri/
└── metadata/
```

---

## 14. Important Points

- Camera images remain raw.
- The PKLs expose camera intrinsics and distortion metadata, but the legacy information-file format does not itself implement a distortion-aware image projection pipeline.
- The final two LiDAR columns are zero placeholders rather than measured point features.
- LiDAR `sweeps` are empty because no non-keyframe LiDAR sweeps were created.
- No radar data is included.
- No map pipeline is available.
- No lidarseg or panoptic labels are generated.
- No ground-truth database is generated.
- The ground-truth database sampler is therefore disabled.
- Test annotations are retained for evaluation and must not be used as training input.

---

## 15. Loader Compatibility

These PKLs target the legacy MMDetection3D/BEVFusion information-file style:

```python
{
    "infos": [...],
    "metadata": {...},
}
```

They do not use the newer MMDetection3D:

```text
data_list / metainfo
```

schema.

Downstream loaders must also account for the AGHRI-specific sensor channels and the documented AGHRI-to-nuScenes-style differences.

For background on the legacy nuScenes information-file workflow, see:

[MMDetection3D nuScenes dataset documentation](https://github.com/open-mmlab/mmdetection3d/blob/1.0/docs/en/datasets/nuscenes_det.md)

---

## 16. Relationship to the Converted JSON Tables

The JSON files under `v1.0-aghri/` describe the relational AGHRI-to-nuScenes-style dataset.

The PKLs reorganise the subset of information needed by a legacy perception loader into one per-sample record containing:

- sensor paths;
- timestamps;
- camera and LiDAR transforms;
- calibration;
- camera model and distortion metadata;
- 3D boxes;
- class names;
- velocities;
- LiDAR point counts; and
- synchronization evidence.

For the source JSON-table and conversion reference, see
[`CONVERTED_DATA_FORMAT.md`](CONVERTED_DATA_FORMAT.md).
