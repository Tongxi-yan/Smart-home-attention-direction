# Smart-Home Attention Direction Detection

This final-year project detects whether a person is attending to neither device, a
desk lamp, or a smart speaker. It covers the complete workflow from Azure Kinect data
collection to real-time GCN-Attention-BiLSTM inference.

## Workflow

```text
Skeleton and RGB collection
          ↓
Lamp/speaker recognition and depth-based 3D localisation
          ↓
Manual attention annotation in ELAN
          ↓
Data merge, interpolation and Gaussian smoothing
          ↓
Feature and temporal-window generation
          ↓
Window-level or segment-level data splitting
          ↓
Five-model training and comparison
          ↓
Real-time inference with the selected model
```

The output classes are `Off`, `Device 1` (Mi Desk Lamp), and `Device 2`
(XiaoAi Speaker).

## Clear module structure

```text
FYP/
├── data_preparation/                 data, merge and smoothing
│   ├── raw_data/                     available original recordings
│   ├── final_data/                   final model-input datasets
│   ├── merge_data.py
│   ├── smooth_data.py
│   └── schema.py
├── device_detection/                lamp/speaker recognition and depth localisation
│   ├── train_device_detector.py
│   ├── locate_devices_with_depth.py
│   ├── device_detector_best.pt
│   └── training_results/
├── source_code/                      reusable research and training code
│   ├── data_collection/
│   ├── feature_generation/
│   ├── data_splitting/
│   └── model_experiments/
├── real_time/                        final model, checkpoint and deployment code
└── tests/                            dependency-light verification
```

There is no additional `source_code/fyp/` level. Technical notes are placed in the
README of the relevant module instead of a separate `docs/` folder.

Experiment evidence is also kept with the code that produced it:

- `device_detection/training_results/` contains the detector curves, confusion
  matrices and training histories.
- `source_code/model_experiments/results/` contains the attention-model comparison.

These compact results show the completed experimental work. Repeated run folders,
caches and generated train/validation/test datasets are excluded.

## Data

The study collected **25 original recordings**. The supplied workspace contains 23
available raw session folders because `Data18` and `Data19` were not present. All 25
final processed datasets are retained; missing raw files have not been fabricated.

Only the start and end of the data pipeline are stored:

- `data_preparation/raw_data/`: skeleton CSVs, 3D device-position CSVs and RGB videos.
- `data_preparation/final_data/`: final aligned, labelled, interpolated and smoothed
  model-input CSVs.

The code needed to reproduce merge and smoothing is retained in the same top-level
module. Intermediate CSVs and model-specific split copies are not stored.

## ELAN annotation

The synchronised RGB recordings were manually annotated in **ELAN** using three
mutually exclusive intervals:

- `Off`: attention is not directed towards either device.
- `On Device 1`: attention is directed towards the lamp.
- `On Device 2`: attention is directed towards the speaker.

An interval starts at the earliest visible intention cue, such as a head turn or hand
movement towards a device, and ends after attention moves away. The ELAN intervals
were aligned to the 30 fps recordings and converted to the frame-level columns
`label_device1`, `label_device2` and `label_3` in the final datasets.

## Module guide

### Data preparation

| File | Purpose |
| --- | --- |
| `data_preparation/merge_data.py` | Aligns raw skeletons, 3D device positions and ELAN-derived labels. |
| `data_preparation/smooth_data.py` | Interpolates missing values, applies Gaussian smoothing and validates output. |
| `data_preparation/schema.py` | Defines joint names, graph nodes, columns and class labels. |

### Data collection and device detection

| File | Purpose |
| --- | --- |
| `source_code/data_collection/collect_skeleton.py` | Records Azure Kinect joints and synchronised RGB video. |
| `source_code/data_collection/kinect_preview.py` | Checks depth, body segmentation and skeleton tracking. |
| `device_detection/train_device_detector.py` | Trains and evaluates the lamp/speaker YOLO segmenter. |
| `device_detection/locate_devices_with_depth.py` | Combines YOLO masks with aligned depth to estimate 3D device positions. |

### Feature generation and splitting

| File or folder | Purpose |
| --- | --- |
| `source_code/feature_generation/build_features.py` | Builds root-relative, distance and motion features. |
| `source_code/feature_generation/create_windows.py` | Creates labelled temporal windows and segment IDs. |
| `source_code/data_splitting/window/` | Randomly splits individual windows. |
| `source_code/data_splitting/segment/` | Keeps every continuous behaviour segment together. |

The segment protocol is stricter because neighbouring windows from one behaviour
segment cannot be placed in different subsets.

### Model experiments

| Folder or file | Purpose |
| --- | --- |
| `source_code/model_experiments/xgboost/` | XGBoost baseline. |
| `source_code/model_experiments/cnn/` | 1D-CNN temporal baseline. |
| `source_code/model_experiments/bilstm/` | Bidirectional LSTM with temporal attention. |
| `source_code/model_experiments/stgcn/` | Spatial-temporal graph convolution baseline. |
| `source_code/model_experiments/gcn_attention_bilstm/` | Selected GCN, joint-attention and BiLSTM model. |
| `source_code/model_experiments/train_models.py` | Shared training entry for all five models. |
| `source_code/model_experiments/evaluation.py` | Accuracy, Balanced Accuracy, Macro F1 and confusion matrices. |
| `source_code/model_experiments/model_config.json` | Window, feature, split, model and real-time settings. |

### Real-time application

| File | Purpose |
| --- | --- |
| `real_time/load_model.py` | Reconstructs the model and loads its checkpoint. |
| `real_time/run_realtime.py` | Performs rolling-window inference, probability smoothing and CSV replay. |
| `real_time/best_gcn_attention_bilstm.pth` | Final attention-classifier checkpoint. |
| `real_time/model_metadata.json` | Stores model provenance, hashes and compatibility details. |

## Installation

Large CSV, video and weight files use Git LFS.

```bash
git lfs install
git clone https://github.com/Tongxi-yan/Smart-home-attention-direction.git
cd Smart-home-attention-direction
git lfs pull

python -m venv .venv
source .venv/bin/activate
pip install -e ".[models]"
```

Kinect and YOLO functions additionally require the Azure Kinect SDK, Body Tracking
SDK and hardware dependencies:

```bash
pip install -e ".[azure]"
```

## Main commands

```bash
# Record a new session
fyp record-skeleton --output data_preparation/raw_data/new_session

# Train the detector and localise devices with depth
fyp train-device-detector --data device_detection/detector_config.yaml --output outputs/device_detection
fyp locate-devices --model device_detection/device_detector_best.pt --output device_positions.csv

# Merge raw sources and produce one final dataset
fyp integrate --skeleton skeleton.csv --devices device_positions.csv --labels elan_labels.csv --output outputs/merged.csv
fyp preprocess --input outputs/merged.csv --output outputs/final.csv

# Train one model using window or segment splitting
fyp train --data data_preparation/final_data --config source_code/model_experiments/model_config.json --model gcn_attention_bilstm --split segment --output outputs/final_model

# Replay the real-time pipeline
fyp realtime --input data_preparation/final_data/dataset_1.csv --checkpoint real_time/best_gcn_attention_bilstm.pth
```

Available models are `xgboost`, `cnn`, `bilstm`, `stgcn` and
`gcn_attention_bilstm`. Available split strategies are `window` and `segment`.
