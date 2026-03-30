"""Optional: compare physics expectations vs ML outputs.

Computes pressure_hat = K_P * flow_rate * temperature and residuals.

Flags a physics_violation when residual exceeds a robust threshold.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

import config as cfg


def check_consistency(
    df: pd.DataFrame,
    *,
    residual_thr_sigma: float = 5.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    out = df.copy()
    if not all(c in out.columns for c in ("temperature", "pressure", "flow_rate")):
        raise ValueError("df must contain temperature, pressure, flow_rate")

    pressure_hat = float(cfg.K_P) * out["flow_rate"].to_numpy(dtype=float) * out["temperature"].to_numpy(dtype=float)
    residual = out["pressure"].to_numpy(dtype=float) - pressure_hat

    # Robust scale via MAD
    med = float(np.median(residual))
    mad = float(np.median(np.abs(residual - med)))
    scale = mad if mad > 0.0 else float(np.std(residual) + 1e-6)
    thr = float(residual_thr_sigma) * float(scale)

    violation = np.abs(residual - med) > thr
    out["pressure_hat"] = pressure_hat
    out["pressure_residual"] = residual
    out["physics_violation_flag"] = violation.astype(int)

    summary = {
        "violation_rate": float(violation.mean()) if violation.size else 0.0,
        "residual_median": float(med),
        "residual_mad": float(mad),
        "threshold": float(thr),
    }
    if "anomaly_flag" in out.columns:
        flagged = (out["anomaly_flag"].to_numpy(dtype=int) != 0)
        summary["agreement_with_anomaly_flag"] = float((flagged == violation).mean()) if violation.size else 0.0
    return out, summary
