"""Unified training entry point for all thesis model families."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from ..config import ProjectConfig, load_config
from ..feature_generation.build_features import window_to_graph
from ..feature_generation.create_windows import WindowDataset, load_window_directory
from ..data_splitting import (
    load_window_keys,
    split_dataset,
    split_with_fixed_test,
)
from .evaluation import calculate_metrics
from .xgboost import build_model as build_xgboost_model
from .xgboost import summarize_windows


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def _class_weights(labels: np.ndarray, num_classes: int = 3) -> np.ndarray:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"Training split is missing a class: counts={counts.tolist()}")
    return (len(labels) / (num_classes * counts)).astype(np.float32)


def _select_split(data: WindowDataset, config: ProjectConfig):
    split_config = config.split
    if split_config.fixed_test_windows:
        return split_with_fixed_test(
            data,
            strategy=split_config.strategy,
            test_keys=load_window_keys(split_config.fixed_test_windows),
            train_ratio=split_config.train_ratio,
            val_ratio=split_config.val_ratio,
            seed=split_config.random_seed,
        )
    return split_dataset(
        data,
        strategy=split_config.strategy,
        ratios=(
            split_config.train_ratio,
            split_config.val_ratio,
            split_config.test_ratio,
        ),
        seed=split_config.random_seed,
    )


def _train_xgboost(
    data: WindowDataset,
    split,
    output_dir: Path,
    config: ProjectConfig,
) -> dict[str, Any]:
    features = summarize_windows(data.X)
    model = build_xgboost_model(random_seed=config.split.random_seed)
    weights = None
    if config.train.class_weighted_loss:
        class_weights = _class_weights(data.y[split.train])
        weights = class_weights[data.y[split.train]]
    model.fit(
        features[split.train],
        data.y[split.train],
        sample_weight=weights,
        eval_set=[(features[split.validation], data.y[split.validation])],
        verbose=False,
    )
    prediction = model.predict(features[split.test]).astype(int)
    metrics = calculate_metrics(data.y[split.test], prediction).to_dict()
    model.save_model(str(output_dir / "best_xgboost.json"))
    _json_dump(output_dir / "metrics.json", metrics)
    return metrics


def _torch_device(requested: str):
    import torch

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_torch_model(config: ProjectConfig, input_size: int, graph_channels: int):
    from .bilstm import BiLSTMClassifier
    from .cnn import CNN1DClassifier
    from .gcn_attention_bilstm import GCNJointAttentionLSTMClassifier
    from .stgcn import STGCNClassifier

    model_config = config.model
    if model_config.architecture == "cnn":
        return CNN1DClassifier(
            input_size=input_size,
            num_classes=3,
            dropout=model_config.dropout,
            classifier_hidden_dim=model_config.classifier_hidden_dim,
        )
    if model_config.architecture == "bilstm":
        return BiLSTMClassifier(
            input_size=input_size,
            hidden_size=model_config.lstm_hidden_dim,
            num_layers=max(model_config.lstm_layers, 1),
            num_classes=3,
            bidirectional=model_config.bidirectional,
            dropout=model_config.dropout,
            classifier_hidden_dim=model_config.classifier_hidden_dim,
        )
    if model_config.architecture == "stgcn":
        return STGCNClassifier(
            in_channels=graph_channels,
            num_classes=3,
            hidden_dims=model_config.gcn_hidden_dims,
            temporal_kernel_sizes=model_config.temporal_kernel_sizes,
            dropout=model_config.dropout,
            classifier_hidden_dim=model_config.classifier_hidden_dim,
        )
    if model_config.architecture == "gcn_attention_bilstm":
        return GCNJointAttentionLSTMClassifier(
            in_channels=graph_channels,
            num_classes=3,
            gcn_hidden_dims=model_config.gcn_hidden_dims,
            attn_hidden_dim=model_config.attention_hidden_dim,
            temporal_kernel_sizes=model_config.temporal_kernel_sizes,
            lstm_hidden_dim=model_config.lstm_hidden_dim,
            lstm_layers=model_config.lstm_layers,
            bidirectional=model_config.bidirectional,
            classifier_hidden_dim=model_config.classifier_hidden_dim,
            dropout=model_config.dropout,
            joint_attention_dropout=model_config.joint_attention_dropout,
            joint_attention_temperature=model_config.joint_attention_temperature,
            temporal_pool=model_config.temporal_pool,
        )
    raise ValueError(f"Unsupported torch architecture: {model_config.architecture}")


def _train_torch(
    data: WindowDataset,
    split,
    output_dir: Path,
    config: ProjectConfig,
) -> dict[str, Any]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError as error:
        raise RuntimeError("Install the model dependencies: pip install -e '.[models]'") from error

    graph_architectures = {"stgcn", "gcn_attention_bilstm"}
    use_graph = config.model.architecture in graph_architectures

    class SequenceDataset(Dataset):
        def __init__(self, indices: np.ndarray) -> None:
            self.indices = np.asarray(indices, dtype=np.int64)

        def __len__(self) -> int:
            return len(self.indices)

        def __getitem__(self, index: int):
            data_index = int(self.indices[index])
            window = data.X[data_index]
            values = window_to_graph(window) if use_graph else window
            return torch.tensor(values), torch.tensor(data.y[data_index], dtype=torch.long)

    device = _torch_device(config.train.device)
    graph_channels = int(window_to_graph(data.X[0]).shape[0])
    model = _build_torch_model(config, data.X.shape[-1], graph_channels).to(device)
    loader_options = {
        "batch_size": config.train.batch_size,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(SequenceDataset(split.train), shuffle=True, **loader_options)
    validation_loader = DataLoader(
        SequenceDataset(split.validation), shuffle=False, **loader_options
    )
    test_loader = DataLoader(SequenceDataset(split.test), shuffle=False, **loader_options)

    weights = None
    if config.train.class_weighted_loss:
        weights = torch.tensor(
            _class_weights(data.y[split.train]), dtype=torch.float32, device=device
        )
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )

    def run(loader, *, training: bool) -> tuple[float, np.ndarray, np.ndarray]:
        model.train(training)
        total_loss = 0.0
        truth: list[np.ndarray] = []
        prediction: list[np.ndarray] = []
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for values, labels in loader:
                values = values.to(device=device, dtype=torch.float32)
                labels = labels.to(device)
                if training:
                    optimizer.zero_grad()
                logits = model(values)
                loss = criterion(logits, labels)
                if training:
                    loss.backward()
                    if config.train.gradient_clip > 0:
                        nn.utils.clip_grad_norm_(
                            model.parameters(), config.train.gradient_clip
                        )
                    optimizer.step()
                total_loss += float(loss.item()) * len(labels)
                truth.append(labels.detach().cpu().numpy())
                prediction.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
        return (
            total_loss / len(loader.dataset),
            np.concatenate(truth),
            np.concatenate(prediction),
        )

    history: list[dict[str, float]] = []
    best_state = None
    best_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, config.train.epochs + 1):
        train_loss, train_true, train_pred = run(train_loader, training=True)
        val_loss, val_true, val_pred = run(validation_loader, training=False)
        train_f1 = calculate_metrics(train_true, train_pred).macro_f1
        val_f1 = calculate_metrics(val_true, val_pred).macro_f1
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_macro_f1": train_f1,
                "validation_loss": val_loss,
                "validation_macro_f1": val_f1,
            }
        )
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.4f} "
            f"train_f1={train_f1:.4f} val_loss={val_loss:.4f} val_f1={val_f1:.4f}"
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.train.patience:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    test_loss, test_true, test_pred = run(test_loader, training=False)
    metrics = calculate_metrics(test_true, test_pred).to_dict()
    metrics.update(
        {"test_loss": test_loss, "best_validation_f1": best_f1, "best_epoch": best_epoch}
    )

    checkpoint = {
        "epoch": best_epoch,
        "model_state_dict": model.state_dict(),
        "best_val_f1": best_f1,
        "model_arch": config.model.architecture,
        "config": {
            "channels_per_node": graph_channels,
            "num_nodes": 16,
            "model": {
                "gcn_hidden_dims": config.model.gcn_hidden_dims,
                "attn_hidden_dim": config.model.attention_hidden_dim,
                "temporal_kernel_sizes": config.model.temporal_kernel_sizes,
                "lstm_hidden_dim": config.model.lstm_hidden_dim,
                "lstm_layers": config.model.lstm_layers,
                "bidirectional": config.model.bidirectional,
                "classifier_hidden_dim": config.model.classifier_hidden_dim,
                "dropout": config.model.dropout,
                "joint_attention_dropout": config.model.joint_attention_dropout,
                "joint_attention_temperature": config.model.joint_attention_temperature,
                "temporal_pool": config.model.temporal_pool,
            },
            "project": config.to_dict(),
        },
    }
    torch.save(checkpoint, output_dir / f"best_{config.model.architecture}.pth")
    _json_dump(output_dir / "history.json", history)
    _json_dump(output_dir / "metrics.json", metrics)
    return metrics


def train(
    data_directory: str | Path,
    output_directory: str | Path,
    config: ProjectConfig,
    *,
    pattern: str = "dataset_*.csv",
) -> dict[str, Any]:
    config.validate()
    _set_seed(config.split.random_seed)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    data = load_window_directory(
        data_directory,
        pattern=pattern,
        window_config=config.windows,
        feature_config=config.features,
    )
    split = _select_split(data, config)
    split_summary = {
        "total": len(data),
        "train": len(split.train),
        "validation": len(split.validation),
        "test": len(split.test),
        "strategy": config.split.strategy,
        "architecture": config.model.architecture,
    }
    _json_dump(output / "run_config.json", config.to_dict())
    _json_dump(output / "split_summary.json", split_summary)
    if config.model.architecture == "xgboost":
        return _train_xgboost(data, split, output, config)
    return _train_torch(data, split, output, config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Directory of processed dataset_*.csv")
    parser.add_argument("--output", required=True, help="Run output directory")
    parser.add_argument("--config", help="JSON configuration; defaults to package settings")
    parser.add_argument(
        "--model",
        choices=("xgboost", "cnn", "bilstm", "stgcn", "gcn_attention_bilstm"),
    )
    parser.add_argument("--split", choices=("window", "segment"))
    parser.add_argument("--pattern", default="dataset_*.csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.model:
        config.model.architecture = args.model
    if args.split:
        config.split.strategy = args.split
    metrics = train(args.data, args.output, config, pattern=args.pattern)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
