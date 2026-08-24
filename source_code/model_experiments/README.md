# Model experiments

Each attempted baseline has one implementation folder:

- `xgboost/`: summary-statistic baseline.
- `cnn/`: 1D temporal CNN.
- `bilstm/`: bidirectional LSTM with temporal attention.
- `stgcn/`: spatial-temporal graph convolution network.
- `gcn_attention_bilstm/`: selected 16-node GCN, joint-attention and BiLSTM model.

`train_models.py` provides the shared training loop for all five models.
`evaluation.py` calculates the common metrics. `model_config.json` stores the
experiment settings, while
`fixed_test_windows.json` stores only reproducibility keys rather than dataset copies.

`results/attention_benchmarks.csv` records the thesis comparison across five models,
window and segment protocols, and window lengths 15, 20 and 30. The best reported
configuration used the GCN-Attention-BiLSTM with a 20-frame input and Macro F1 0.977.
