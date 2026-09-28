# AGHRI-to-nuScenes-Style Conversion and JSON Reference

This document describes the output of the **AGHRI-to-nuScenes-style conversion**, including the converted sensor files, relational metadata tables, coordinate conventions, camera models, annotations, dataset splits, and synchronization metadata.

The conversion uses the nuScenes relational data model as a structural reference while preserving documented AGHRI-specific choices. The output should therefore be treated as **AGHRI data represented using a nuScenes-style relational structure**, not as an official nuScenes dataset.

---

## 1. Conversion Overview

The original AGHRI release contains sensor files, annotations, calibration, odometry, and TF information organised by recording.

The conversion reorganises these inputs into:

```text
Original AGHRI recording
        │
        ▼
validated LiDAR annotation/keyframe selection
        │
        ├── camera–LiDAR association
        ├── calibration and pose handling
        ├── point-cloud conversion
        └── 3D annotation conversion
        │
        ▼
AGHRI-to-nuScenes-style dataset
        │
        ├── samples/
        ├── sweeps/
        ├── v1.0-aghri/*.json
        └── metadata/synchronization/
```

Only validated annotated LiDAR frames become LiDAR keyframe samples. Camera frames associated with those anchors become camera keyframes; additional accepted camera frames can be represented as camera sweeps.

No non-keyframe LiDAR sweep is created.

---

## 2. Output Directory Layout

```text
AGHRI_nuScenes_camera_lidar_full/
├── samples/
│   ├── lidar/
│   ├── cam_zed_rgb/
│   ├── cam_fish_front/
│   ├── cam_fish_left/
│   └── cam_fish_right/
├── sweeps/
│   ├── cam_zed_rgb/
│   ├── cam_fish_front/
│   ├── cam_fish_left/
│   └── cam_fish_right/
├── v1.0-aghri/
│   ├── category.json
│   ├── attribute.json
│   ├── visibility.json
│   ├── sensor.json
│   ├── calibrated_sensor.json
│   ├── ego_pose.json
│   ├── log.json
│   ├── scene.json
│   ├── sample.json
│   ├── sample_data.json
│   ├── instance.json
│   ├── sample_annotation.json
│   ├── map.json
│   ├── splits.json
│   └── camera_model.json
└── metadata/
    └── synchronization/
        ├── aghri-scene-0001_sync.json
        └── ...
```

The `metadata/synchronization/` files are runtime metadata used by the PKL generator. They are part of the converted representation rather than optional diagnostic reports.

---

## 3. Converted Dataset Scope

The completed conversion contains:

| Item | Converted AGHRI release |
|---|---:|
| Scenes | 65 |
| Train / validation / test scenes | 52 / 7 / 6 |
| LiDAR-anchored samples | 13,605 |
| `sample_data` records | 141,305 |
| 3D human annotations | 26,637 |
| Tracked human instances | 136 |

Active sensor channels are:

```text
lidar
cam_zed_rgb
cam_fish_front
cam_fish_left
cam_fish_right
```

ZED depth, radar, lidarseg, panoptic labels, and a map representation are not included in this conversion.

`attribute.json`, `visibility.json`, and `map.json` are empty arrays in the released conversion.

---

## 4. Relational Structure

The AGHRI metadata is organised using the same general relational idea as nuScenes: the large camera and LiDAR files are stored separately, while JSON tables describe how recordings, samples, sensors, poses, and annotations relate to each other.

### Recording and sensor side

```text
log
 └── scene
      └── sample
           └── sample_data
                ├── calibrated_sensor
                │    └── sensor
                └── ego_pose
```

### Annotation side

```text
sample
 └── sample_annotation
      └── instance
           └── category
```

`attribute.json` and `visibility.json` are retained for structural compatibility but are empty in this AGHRI-to-nuScenes-style release.

---

## 5. JSON Table Reference

### 5.1 `category.json`

Purpose: defines the detection category used by the converted 3D annotations.

The conversion contains the single detection class:

```text
human
```

This is a 3D detection category table, not a semantic-segmentation class list.

Typical relationship:

```text
instance.category_token → category.token
```

---

### 5.2 `attribute.json`

Purpose in the nuScenes relational model: optional temporary object states.

AGHRI does not provide equivalent validated per-frame detection attributes for this conversion, so:

```json
[]
```

is used.

No downstream method should assume moving/standing/sitting attributes are available from this table.

---

### 5.3 `visibility.json`

Purpose in the nuScenes relational model: links an annotation to a visibility level.

The AGHRI-to-nuScenes-style 3D detection conversion does not populate this table, so:

```json
[]
```

is used.

---

### 5.4 `sensor.json`

Purpose: defines each active sensor channel and modality.

The converted channels are:

| Channel | Modality |
|---|---|
| `lidar` | LiDAR |
| `cam_zed_rgb` | camera |
| `cam_fish_front` | camera |
| `cam_fish_left` | camera |
| `cam_fish_right` | camera |

A `calibrated_sensor` record links back to the corresponding sensor definition.

---

### 5.5 `calibrated_sensor.json`

Purpose: records how a sensor is mounted relative to the AGHRI robot ego frame (`base_link`).

Conceptually:

```text
sensor frame → AGHRI base_link ego frame
```

The table stores sensor translation and rotation and, for cameras, the intrinsic matrix required by downstream projection code.

Quaternions use:

```text
[w, x, y, z]
```

The intrinsic matrix does not by itself describe lens distortion. AGHRI camera model and distortion information is stored separately in `camera_model.json`.

---

### 5.6 `camera_model.json`

Purpose: preserves AGHRI camera-model information that is not represented by a standard intrinsic matrix alone.

For each camera calibration, this table records information including:

- camera model;
- distortion model;
- distortion coefficients;
- distortion coefficient order;
- raw image width and height;
- image state;
- rectification matrix;
- projection matrix; and
- calibration source.

#### ZED RGB

The ZED RGB calibration is represented with:

- projection geometry compatible with a pinhole intrinsic matrix;
- source distortion model `rational_polynomial`;
- the corresponding distortion coefficients.

Calling the projection geometry `pinhole` does **not** mean the stored raw image is distortion-free.

#### Fisheye cameras

The front, left, and right fisheye cameras retain:

- fisheye camera geometry;
- distortion model `equidistant`;
- four distortion coefficients.

#### Image state

All released camera images remain:

```text
raw
```

The conversion does not rectify or undistort them.

---

### 5.7 `ego_pose.json`

Purpose: records the robot ego pose at the timestamp of an individual sensor measurement.

Conceptually:

```text
AGHRI base_link ego frame → sequence-local global frame
```

Pose interpolation uses the AGHRI pose/TF information, including:

```text
metadata/odom_global.jsonl
metadata/tf/map__to__odom.jsonl
metadata/tf/odom__to__base_link.jsonl
```

Measurements that cannot be safely interpolated within the configured bracket limits are excluded rather than extrapolated.

The released global frame is sequence-local and rebased at the first accepted pose.

Quaternions use:

```text
[w, x, y, z]
```

---

### 5.8 `log.json`

Purpose: records the broader AGHRI recording context associated with a scene.

The table provides a stable log token and recording context that can include:

- recording identifier;
- robot/platform identifier;
- capture date; and
- location/environment description.

A scene links to its log through `log_token`.

---

### 5.9 `scene.json`

Purpose: represents one converted AGHRI recording/sequence.

Important fields follow the relational scene concept:

| Field | Meaning |
|---|---|
| `token` | unique scene identifier |
| `log_token` | link to the parent recording/log |
| `nbr_samples` | number of LiDAR-anchored key samples in the scene |
| `first_sample_token` | first sample in the scene |
| `last_sample_token` | last sample in the scene |
| `name` | public scene name such as `aghri-scene-0001` |
| `description` | human-readable scene description |

The conversion contains 65 scenes.

---

### 5.10 `sample.json`

Purpose: represents one LiDAR-anchored key moment in a converted AGHRI scene.

Important fields include:

| Field | Meaning |
|---|---|
| `token` | unique sample identifier |
| `timestamp` | LiDAR anchor timestamp in microseconds |
| `scene_token` | scene containing the sample |
| `prev` | previous sample in the same scene |
| `next` | next sample in the same scene |

The sample is a logical key moment. Actual sensor files are represented through `sample_data.json`.

---

### 5.11 `sample_data.json`

Purpose: represents one physical sensor capture and points to its stored file.

Important fields include:

| Field | Meaning |
|---|---|
| `token` | unique sensor-capture identifier |
| `sample_token` | logical key sample associated with the capture |
| `ego_pose_token` | robot pose at the sensor timestamp |
| `calibrated_sensor_token` | calibration used by the sensor |
| `timestamp` | actual sensor timestamp in microseconds |
| `fileformat` | stored file format |
| `is_key_frame` | whether this capture is used as a keyframe |
| `height`, `width` | image dimensions; zero for point clouds |
| `filename` | path relative to the converted dataset root |
| `prev`, `next` | neighbouring captures from the same sensor stream |

Examples of relative paths include:

```text
samples/lidar/...
samples/cam_zed_rgb/...
samples/cam_fish_front/...
sweeps/cam_fish_left/...
```

---

### 5.12 `instance.json`

Purpose: represents one physical human tracked through a scene.

Important relationships and fields include:

| Field | Meaning |
|---|---|
| `token` | unique tracked-instance identifier |
| `category_token` | link to the `human` detection category |
| `nbr_annotations` | number of 3D annotation records in the track |
| `first_annotation_token` | first annotation for the tracked human |
| `last_annotation_token` | last annotation for the tracked human |

The completed conversion contains 136 tracked human instances.

---

### 5.13 `sample_annotation.json`

Purpose: stores one 3D human bounding box for one sample.

Important fields include:

| Field | Meaning |
|---|---|
| `token` | annotation identifier |
| `sample_token` | sample containing the annotation |
| `instance_token` | tracked human represented by the annotation |
| `translation` | box centre in the sequence-local global frame |
| `size` | `[width, length, height]` in metres |
| `rotation` | box orientation in the global frame as `[w, x, y, z]` |
| `prev`, `next` | neighbouring annotations of the same instance |
| `num_lidar_pts` | number of LiDAR points inside the box |
| `num_radar_pts` | always `0` for this release |

The source LiDAR boxes are treated as axis-aligned according to the conversion policy. Their size fields are mapped into width/length/height order.

A released global annotation quaternion can be non-zero because an identity-orientation source box is transformed through the moving ego pose into the sequence-local global frame.

---

### 5.14 `map.json`

AGHRI does not provide the map representation required by the standard nuScenes map pipeline.

The released table is therefore:

```json
[]
```

Downstream code must not assume a map record is available.

---

### 5.15 `splits.json`

`split.json` is not used; the AGHRI-specific table is:

```text
splits.json
```

Purpose: records the fixed sequence-level AGHRI partition used by the conversion and PKL generator.

The split contains:

- 52 training scenes;
- 7 validation scenes;
- 6 test scenes.

The table is generated from the split text files under:

```text
splits/train.txt
splits/val.txt
splits/test.txt
```

The split is sequence-level rather than frame-level.

---

## 6. LiDAR Representation

Original AGHRI point clouds are converted to little-endian float32 binary files with five columns:

```text
[x, y, z, 0, 0]
```

The final two values are placeholders.

They are **not** measured:

- LiDAR intensity;
- ring index; or
- point time.

Coordinate convention:

```text
x = forward
y = left
z = up
```

Coordinates are in metres.

Only validated annotated LiDAR frames become LiDAR keyframes. No additional LiDAR sweeps are created.

---

## 7. Camera Representation

Camera images are copied without changing their encoded image bytes.

The conversion does not:

- undistort images;
- rectify images; or
- resample images.

The released images therefore remain the original raw AGHRI images.

The camera intrinsic matrix, camera model, distortion model, and distortion coefficients are stored as metadata so that downstream methods can decide whether to use a pinhole approximation or a distortion-aware projection.

---

## 8. Poses and Coordinate Frames

The relevant conceptual frames are:

```text
sequence-local global frame
        ↑
      ego pose
        ↑
AGHRI base_link
        ↑
sensor calibration
        ↑
camera / LiDAR frame
```

Sensor extrinsics are represented relative to AGHRI `base_link`.

Ego poses are evaluated at the actual sensor timestamps.

The sequence-local global frame is rebased at the first accepted pose for each converted sequence.

---

## 9. Samples and Sweeps

In this AGHRI-to-nuScenes-style representation:

- `samples/` contains main sensor captures associated with key samples;
- `sweeps/` contains additional accepted camera captures between keyframes;
- LiDAR keyframes are in `samples/lidar/`;
- no non-keyframe LiDAR sweep is generated.

Camera `sample_data` records can therefore be keyframes or sweeps, while LiDAR is keyframe-only in this conversion.

---

## 10. Camera–LiDAR Synchronization

LiDAR is the anchor stream.

For each AGHRI recording and camera stream, the conversion:

1. evaluates the camera timestamps relative to LiDAR anchors;
2. estimates the configured per-camera timing offset;
3. aligns camera times using that offset;
4. applies the accepted per-recording timing threshold; and
5. stores either the accepted camera association or the rejection information.

Per-scene results are written to:

```text
metadata/synchronization/aghri-scene-XXXX_sync.json
```

These files store information such as:

- LiDAR anchor;
- selected camera associations;
- estimated per-camera constant offsets;
- aligned timestamps;
- synchronization residuals;
- rejection reasons; and
- the acceptance threshold used for that recording.

The PKL generator reads these associations directly and does not rematch timestamps.

Thresholds above 0.10 s are fallback acceptance limits and must not be described as 50 ms synchronization accuracy.

---

## 11. Compatibility Boundary

The output follows the nuScenes relational data model where practical, but it remains an **AGHRI-to-nuScenes-style representation**.

Important AGHRI-specific differences include:

- AGHRI sensor channel names;
- one LiDAR and four camera channels rather than the nuScenes sensor suite;
- raw ZED and fisheye images with explicit distortion metadata;
- `camera_model.json`;
- `splits.json`;
- an empty `map.json`;
- empty `attribute.json` and `visibility.json`;
- no radar;
- no lidarseg or panoptic labels;
- no non-keyframe LiDAR sweeps.

Downstream consumers must explicitly support these differences.

---

## 12. Relationship to PKL Generation

The PKL generator uses:

```text
v1.0-aghri/*.json
samples/
sweeps/
metadata/synchronization/
```

to build the legacy information PKLs.

The JSON tables describe the relational dataset structure; the PKLs reorganise the required paths, transforms, annotations, and camera information into a loader-oriented catalogue.

See [`PKL_FORMAT.md`](PKL_FORMAT.md) for the PKL-generation and field reference.
