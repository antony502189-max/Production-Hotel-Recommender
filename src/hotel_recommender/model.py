from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .domain import FEATURE_NAMES


def build_ranker(random_state: int = 42) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=220,
        max_leaf_nodes=31,
        min_samples_leaf=15,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=random_state,
    )


def fit_ranker(
    training_frame: pd.DataFrame, random_state: int = 42
) -> HistGradientBoostingClassifier:
    required = set(FEATURE_NAMES) | {"target"}
    missing = required.difference(training_frame.columns)
    if missing:
        raise ValueError(f"training frame is missing columns: {sorted(missing)}")
    if training_frame["target"].nunique() < 2:
        raise ValueError("training data must contain both positive and negative examples")

    model = build_ranker(random_state=random_state)
    model.fit(training_frame.loc[:, list(FEATURE_NAMES)], training_frame["target"].astype(int))
    return model


def predict_scores(model: HistGradientBoostingClassifier, features: pd.DataFrame) -> np.ndarray:
    if features.empty:
        return np.array([], dtype=float)
    return np.asarray(model.predict_proba(features.loc[:, list(FEATURE_NAMES)])[:, 1], dtype=float)
