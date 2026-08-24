"""XGBoost baseline and its fixed window-summary features."""

from __future__ import annotations

import numpy as np


def summarize_windows(windows: np.ndarray) -> np.ndarray:
    """Return mean, standard deviation, and end-to-start change per window."""

    return np.concatenate(
        (
            windows.mean(axis=1),
            windows.std(axis=1),
            windows[:, -1] - windows[:, 0],
        ),
        axis=1,
    ).astype(np.float32)


def build_model(*, random_seed: int):
    try:
        from xgboost import XGBClassifier
    except ImportError as error:
        raise RuntimeError(
            "Install the model dependencies: pip install -e '.[models]'"
        ) from error

    return XGBClassifier(
        n_estimators=2000,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=random_seed,
        n_jobs=-1,
    )
