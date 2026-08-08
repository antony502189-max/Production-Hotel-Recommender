from __future__ import annotations

from pathlib import Path

import joblib

from .domain import FEATURE_NAMES, RecommenderBundle


def load_bundle(path: str | Path) -> RecommenderBundle:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"model artifact not found: {source}")
    bundle = joblib.load(source)
    if not isinstance(bundle, RecommenderBundle):
        raise TypeError("artifact is not a RecommenderBundle")
    if tuple(bundle.metadata.feature_names) != FEATURE_NAMES:
        raise ValueError("model feature schema is incompatible with this application version")
    return bundle
