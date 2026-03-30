"""Real-time anomaly scoring, feature engineering, and status mapping."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd

import config as cfg
from features import make_features


def load_model(model_path: str) -> Dict[str, Any]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(path)


def score_and_flag(df: pd.DataFrame, model_bundle: Dict[str, Any], thresholds: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    out = make_features(df=df, window=10, eps=1e-6)

    feature_order = model_bundle["feature_order"]
    model = model_bundle["model"]
    X = out[feature_order].to_numpy(dtype=float)

    # IsolationForest decision_function: higher means more normal.
    scores = model.decision_function(X)
    out["anomaly_score"] = scores

    warning_thr = float(cfg.WARNING_SCORE_THRESHOLD)
    critical_thr = float(cfg.CRITICAL_SCORE_THRESHOLD)
    if thresholds is not None:
        # Optional override (adaptive thresholds). Must preserve critical < warning.
        warning_thr = float(thresholds["warning"])
        critical_thr = float(thresholds["critical"])

    def _status(score: float) -> str:
        if score < float(critical_thr):
            return "CRITICAL"
        if score < float(warning_thr):
            return "WARNING"
        return "NORMAL"

    out["status"] = out["anomaly_score"].apply(_status)
    out["anomaly_flag"] = (out["status"] != "NORMAL").astype(int)

    # Multi-fault classification (adds a new column; anomaly logic unchanged)
    out["fault_class"] = out.apply(
        lambda r: classify_fault(r["temperature"], r["pressure"], r["flow_rate"]),
        axis=1,
    )
    return out


def classify_fault(T: float, P: float, F: float) -> str:
    # classification priority (required): overheating > pump_failure > pressure_spike
    if float(T) > float(cfg.TEMP_HIGH_THRESHOLD):
        return "overheating"
    if float(F) < float(cfg.FLOW_LOW_THRESHOLD):
        return "pump_failure"
    if (
        float(P) > float(cfg.PRESSURE_HIGH_THRESHOLD)
        and float(T) <= float(cfg.TEMP_HIGH_THRESHOLD)
        and float(F) >= float(cfg.FLOW_LOW_THRESHOLD)
    ):
        return "pressure_spike"
    return "none"
