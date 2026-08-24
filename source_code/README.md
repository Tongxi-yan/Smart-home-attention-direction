# Source code

This folder contains the reusable research code. It has no additional `fyp/` wrapper.

```text
source_code/
├── data_collection/       Azure Kinect skeleton and RGB collection
├── feature_generation/    engineered features and temporal windows
├── data_splitting/        separate window and segment protocols
├── model_experiments/     four baselines plus shared five-model training
├── cli.py                 command dispatcher
└── config.py              validated project settings
```

Data merge and smoothing are in the top-level `data_preparation/` module. Device
recognition and depth localisation are in `device_detection/`. The selected final
model and deployment code are in `real_time/`.

The small `__init__.py` files are Python package markers, not additional experiments.
