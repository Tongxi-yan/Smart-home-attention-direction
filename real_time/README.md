# Real-time application

This standalone folder contains the deployment files for the selected attention model:

- `load_model.py`: checkpoint reconstruction and strict weight loading.
- `run_realtime.py`: 20-frame rolling buffer, stride-based inference, probability
  smoothing, confidence gating and CSV replay.
- `best_gcn_attention_bilstm.pth`: final classifier checkpoint.
- `model_metadata.json`: checkpoint hash, dimensions and provenance.

The model architecture is defined once in
`source_code/model_experiments/gcn_attention_bilstm/model.py`. Shared feature and schema
code is imported from `source_code/feature_generation/` and
`data_preparation/schema.py`, so deployment code is not duplicated. The separate
lamp/speaker detector and depth localisation stage is in `device_detection/`.

Run from the repository root:

```bash
fyp realtime \
  --input data_preparation/final_data/dataset_1.csv \
  --checkpoint real_time/best_gcn_attention_bilstm.pth
```
