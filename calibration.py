"""Calibration utilities for anomaly score distribution."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def analyze_score_distribution(df_scored: pd.DataFrame) -> Dict[str, Any]:
    if df_scored is None or df_scored.empty:
        return {"n": 0, "quantiles": {}, "histogram": {"bins": [], "counts": []}}
    if "anomaly_score" not in df_scored.columns:
        raise ValueError("df_scored must include anomaly_score")

    scores = df_scored["anomaly_score"].to_numpy(dtype=float)
    qs = [0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99]
    quant = {f"p{int(q*100)}": float(np.quantile(scores, q)) for q in qs}

    counts, edges = np.histogram(scores, bins=30)
    return {
        "n": int(scores.size),
        "quantiles": quant,
        "histogram": {"bins": [float(x) for x in edges.tolist()], "counts": [int(c) for c in counts.tolist()]},
    }
