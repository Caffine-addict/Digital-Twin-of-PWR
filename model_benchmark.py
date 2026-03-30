"""Optional: compare anomaly detection methods.

Lightweight and deterministic.

Methods:
- baseline: existing IsolationForest anomaly_score/status from score_and_flag
- zscore: rule-based threshold on z-score of (temperature, pressure, flow_rate)
- mad: robust rule-based threshold on MAD-based z
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

import config as cfg
from features import make_features
from predict import score_and_flag


def _zscore_flags(x: np.ndarray, thr: float = 3.0) -> np.ndarray:
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd == 0.0, 1.0, sd)
    z = (x - mu) / sd
    return (np.abs(z) > float(thr)).any(axis=1)


def _mad_flags(x: np.ndarray, thr: float = 3.5) -> np.ndarray:
    med = np.median(x, axis=0)
    mad = np.median(np.abs(x - med), axis=0)
    mad = np.where(mad == 0.0, 1.0, mad)
    rz = 0.6745 * (x - med) / mad
    return (np.abs(rz) > float(thr)).any(axis=1)


def benchmark(
    df: pd.DataFrame,
    *,
    model_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    # Baseline
    base = score_and_flag(df, model_bundle)
    base_flag = (base["anomaly_flag"].to_numpy(dtype=int) != 0)

    # Rule methods
    feat = make_features(df=df, window=10, eps=1e-6)
    X = feat[["temperature", "pressure", "flow_rate"]].to_numpy(dtype=float)
    z_flag = _zscore_flags(X)
    mad_flag = _mad_flags(X)

    def rates(flag: np.ndarray) -> Dict[str, float]:
        return {
            "alert_rate": float(flag.mean()) if flag.size else 0.0,
            "agreement_with_baseline": float((flag == base_flag).mean()) if flag.size else 0.0,
        }

    report = {
        "baseline": {
            "warning_threshold": float(cfg.WARNING_SCORE_THRESHOLD),
            "critical_threshold": float(cfg.CRITICAL_SCORE_THRESHOLD),
            "alert_rate": float(base_flag.mean()) if base_flag.size else 0.0,
        },
        "zscore": rates(z_flag),
        "mad": rates(mad_flag),
        "n_points": int(len(df)),
    }
    return report
