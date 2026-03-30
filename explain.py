"""Rule-based anomaly explanations using engineered features.

Strict requirements:
- MUST use features from features.make_features
- NO heavy libraries
- Deterministic
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Union

import numpy as np
import pandas as pd

import config as cfg
from features import make_features


RowLike = Union[Mapping[str, Any], pd.Series]


def _get_float(row: RowLike, key: str, default: float = 0.0) -> float:
    try:
        v = row[key]  # type: ignore[index]
    except Exception:
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


def explain_anomaly(row: RowLike) -> Dict[str, Any]:
    """Explain a single row of engineered features."""

    T = _get_float(row, "temperature", 0.0)
    P = _get_float(row, "pressure", 0.0)
    F = _get_float(row, "flow_rate", 0.0)
    dT = _get_float(row, "dT", 0.0)
    dF = _get_float(row, "dF", 0.0)

    # Derivative thresholds are deterministic and tied to SIGMA.
    dT_high_thr = 3.0 * float(cfg.SIGMA)
    dF_drop_thr = -3.0 * float(cfg.SIGMA)

    contributing: List[str] = []
    satisfied = 0
    total_checks = 5

    # Overheating
    temp_high = T > float(cfg.TEMP_HIGH_THRESHOLD)
    dt_high = dT > float(dT_high_thr)
    if temp_high:
        contributing.append("temperature_high")
        satisfied += 1
    if dt_high:
        contributing.append("dT_high")
        satisfied += 1

    overheating = temp_high or dt_high

    # Pump failure
    flow_low = F < float(cfg.FLOW_LOW_THRESHOLD)
    flow_dropping = dF < float(dF_drop_thr)
    if flow_low:
        contributing.append("flow_rate_low")
        satisfied += 1
    if flow_dropping:
        contributing.append("flow_rate_dropping")
        satisfied += 1

    pump_failure = flow_low or flow_dropping

    # Pressure spike (high pressure with normal flow+temp)
    pressure_spike = (
        P > float(cfg.PRESSURE_HIGH_THRESHOLD)
        and T <= float(cfg.TEMP_HIGH_THRESHOLD)
        and F >= float(cfg.FLOW_LOW_THRESHOLD)
    )
    if pressure_spike:
        contributing.append("pressure_high_with_normal_temp_and_flow")
        satisfied += 1

    # Primary cause (priority)
    if overheating:
        primary = "overheating"
    elif pump_failure:
        primary = "pump_failure"
    elif pressure_spike:
        primary = "pressure_spike"
    else:
        primary = "unknown"

    conf = float(satisfied) / float(total_checks)
    conf = float(np.clip(conf, 0.0, 1.0))

    return {
        "primary_cause": primary,
        "contributing_factors": contributing,
        "confidence": conf,
    }


def explain_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply explain_anomaly row-wise after computing engineered features."""
    feat = make_features(df=df, window=10, eps=1e-6)
    primary = []
    factors = []
    confs = []
    for _, row in feat.iterrows():
        e = explain_anomaly(row)
        primary.append(e["primary_cause"])
        factors.append(e["contributing_factors"])
        confs.append(float(e["confidence"]))
    out = feat.copy()
    out["primary_cause"] = primary
    out["contributing_factors"] = factors
    out["confidence"] = confs
    return out
