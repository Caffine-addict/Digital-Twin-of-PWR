"""Real-time anomaly scoring, feature engineering, and status mapping."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

from features import make_features


def load_model(model_path: str) -> Dict[str, Any]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(path)


def score_and_flag(df: pd.DataFrame, model_bundle: Dict[str, Any]) -> pd.DataFrame:
    out = make_features(df=df, window=10, eps=1e-6)

    feature_order = model_bundle["feature_order"]
    model = model_bundle["model"]
    X = out[feature_order].to_numpy(dtype=float)

    # IsolationForest decision_function: higher means more normal.
    scores = model.decision_function(X)
    out["anomaly_score"] = scores

    def _status(score: float) -> str:
        if score < -0.2:
            return "CRITICAL"
        if score < -0.1:
            return "WARNING"
        return "NORMAL"

    out["status"] = out["anomaly_score"].apply(_status)
    out["anomaly_flag"] = (out["status"] != "NORMAL").astype(int)
    return out
