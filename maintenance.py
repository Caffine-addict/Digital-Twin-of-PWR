from __future__ import annotations

from typing import Any, Dict
import numpy as np

import config as cfg

WINDOW = 10


def _slope_last_n(y: np.ndarray) -> float:
    t = np.arange(len(y), dtype=float)
    A = np.column_stack([t, np.ones_like(t)])
    a, _b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a)


def compute_health_metrics(data: Any) -> Dict[str, Any]:
    """
    data: DataFrame-like with columns temperature, flow_rate
    returns:
      { "risk_score": float in [0,1], "maintenance_flag": bool }
    """
    T = data["temperature"].tail(WINDOW).to_numpy(dtype=float)
    F = data["flow_rate"].tail(WINDOW).to_numpy(dtype=float)
    if len(T) < WINDOW or len(F) < WINDOW:
        raise ValueError("need at least 10 points to compute health metrics")

    # rolling mean (computed for completeness; not required in output)
    _T_mean = float(np.mean(T))
    _F_mean = float(np.mean(F))
    _ = (_T_mean, _F_mean)

    temp_slope = _slope_last_n(T)  # increasing => risk
    flow_slope = _slope_last_n(F)  # decreasing => risk

    temp_contrib_raw = max(0.0, temp_slope) / float(cfg.MAINT_TEMP_SLOPE_SCALE)
    flow_contrib_raw = max(0.0, -flow_slope) / float(cfg.MAINT_FLOW_SLOPE_SCALE)

    # clamp contributions using min(1.0, value) as required
    temp_contrib = min(1.0, float(temp_contrib_raw))
    flow_contrib = min(1.0, float(flow_contrib_raw))

    risk_score = 0.5 * temp_contrib + 0.5 * flow_contrib
    risk_score = float(np.clip(risk_score, 0.0, 1.0))

    return {
        "risk_score": risk_score,
        "maintenance_flag": bool(risk_score >= float(cfg.MAINT_RISK_FLAG_THRESHOLD)),
    }
