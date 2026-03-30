"""Evaluation metrics for digital twin anomaly detection and prediction.

Lightweight:
- numpy + pandas only
Deterministic:
- purely functional computations

Metrics:
- false_positive_rate
- detection_rate
- detection_delay
- prediction_mae
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _as_bool_array(x: Any, *, n: int) -> np.ndarray:
    if isinstance(x, np.ndarray):
        arr = x
    else:
        arr = np.asarray(x)
    if arr.shape[0] != n:
        raise ValueError(f"expected array length {n}, got {arr.shape[0]}")
    return arr.astype(bool)


def compute_false_positive_rate(*, y_true: Sequence[bool], y_pred: Sequence[bool]) -> float:
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have same shape")
    neg = ~y_true
    denom = int(neg.sum())
    if denom == 0:
        return 0.0
    fp = int((y_pred & neg).sum())
    return float(fp / denom)


def compute_detection_rate(*, y_true: Sequence[bool], y_pred: Sequence[bool]) -> float:
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have same shape")
    pos = y_true
    denom = int(pos.sum())
    if denom == 0:
        return 0.0
    tp = int((y_pred & pos).sum())
    return float(tp / denom)


def compute_detection_delay(
    *,
    time: Sequence[float],
    y_true: Sequence[bool],
    y_pred: Sequence[bool],
) -> Optional[float]:
    """Detection delay in time units (same as `time`).

    Returns:
    - None if there is no true event window or no detection.
    """
    t = np.asarray(time, dtype=float)
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    if not (t.shape == y_true.shape == y_pred.shape):
        raise ValueError("time, y_true, y_pred must have same shape")

    true_idx = np.where(y_true)[0]
    if true_idx.size == 0:
        return None
    start_i = int(true_idx[0])
    pred_idx = np.where(y_pred & (np.arange(len(y_pred)) >= start_i))[0]
    if pred_idx.size == 0:
        return None
    delay = float(t[int(pred_idx[0])] - t[start_i])
    return delay


def compute_prediction_mae(*, y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have same shape")
    if yt.size == 0:
        return 0.0
    return float(np.mean(np.abs(yt - yp)))


def fault_window_from_schedule(
    *,
    time: Sequence[float],
    fault_schedule: Optional[List[Dict[str, Any]]],
) -> np.ndarray:
    """Create a boolean ground-truth window from a fault schedule."""
    t = np.asarray(time, dtype=float)
    y = np.zeros_like(t, dtype=bool)
    if not fault_schedule:
        return y
    for f in fault_schedule:
        start = float(f.get("start_time", 0.0))
        end = float(f.get("end_time", -1.0))
        y |= (t >= start) & (t <= end)
    return y


def compute_all_metrics(
    *,
    df_scored: pd.DataFrame,
    fault_schedule: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Compute all detection metrics from a scored dataframe."""
    if df_scored is None or df_scored.empty:
        return {
            "false_positive_rate": 0.0,
            "detection_rate": 0.0,
            "detection_delay": None,
        }

    if "time" not in df_scored.columns or "anomaly_flag" not in df_scored.columns:
        raise ValueError("df_scored must include time and anomaly_flag")

    time = df_scored["time"].to_numpy(dtype=float)
    y_pred = (df_scored["anomaly_flag"].to_numpy(dtype=int) != 0)
    y_true = fault_window_from_schedule(time=time, fault_schedule=fault_schedule)

    return {
        "false_positive_rate": compute_false_positive_rate(y_true=y_true, y_pred=y_pred),
        "detection_rate": compute_detection_rate(y_true=y_true, y_pred=y_pred),
        "detection_delay": compute_detection_delay(time=time, y_true=y_true, y_pred=y_pred),
    }