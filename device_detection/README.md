# Device detection and depth localisation

This module contains the complete lamp/speaker recognition stage:

- `train_device_detector.py`: two-stage YOLO segmentation training and evaluation.
- `locate_devices_with_depth.py`: YOLO mask inference, aligned depth lookup, robust 3D
  median estimation and temporal smoothing.
- `device_detector_best.pt`: selected lamp/speaker detector checkpoint.
- `detector_config.yaml`: detector dataset configuration.
- `training_results/`: compact training histories, curves and confusion matrices.

The training results are retained as evidence of the completed detector experiments;
large repeated run folders are not included.

The original Roboflow export is not stored because it is already divided into
train/validation/test subsets. To retrain, place it at
`data_preparation/device_detection_dataset/`, then run:

```bash
fyp train-device-detector \
  --data device_detection/detector_config.yaml \
  --output outputs/device_detection
```

To collect lamp and speaker positions using aligned depth:

```bash
fyp locate-devices \
  --model device_detection/device_detector_best.pt \
  --output device_positions.csv
```
