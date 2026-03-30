"""Optional: simple Kalman filter state estimator.

Estimates true state from noisy measurements for (temperature, pressure, flow_rate).

Lightweight:
- numpy + pandas only
Deterministic:
- no randomness
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple, Optional

import numpy as np
import pandas as pd


STATE_COLS_DEFAULT = ("temperature", "pressure", "flow_rate")


@dataclass(frozen=True)
class KalmanConfig:
    q_diag: Tuple[float, float, float] = (0.05, 0.05, 0.05)
    r_diag: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    p0_diag: Tuple[float, float, float] = (1.0, 1.0, 1.0)


def _as_diag(diag3: Sequence[float]) -> np.ndarray:
    d = np.asarray(diag3, dtype=float)
    if d.shape != (3,):
        raise ValueError("expected diag length 3")
    return np.diag(d)


def kalman_filter(
    df: pd.DataFrame,
    *,
    cols: Sequence[str] = STATE_COLS_DEFAULT,
    config: KalmanConfig = KalmanConfig(),
    x0: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    if any(c not in df.columns for c in cols):
        raise ValueError(f"missing required measurement columns: {cols}")

    z = df[list(cols)].to_numpy(dtype=float)
    n = z.shape[0]
    out = df.copy()
    if n == 0:
        for c in cols:
            out[f"{c}_est"] = np.nan
        return out

    F = np.eye(3, dtype=float)
    H = np.eye(3, dtype=float)
    Q = _as_diag(config.q_diag)
    R = _as_diag(config.r_diag)
    P = _as_diag(config.p0_diag)
    I = np.eye(3, dtype=float)

    x = np.asarray(x0, dtype=float) if x0 is not None else z[0].copy()
    if x.shape != (3,):
        raise ValueError("x0 must be length 3")

    x_hist = np.zeros((n, 3), dtype=float)

    for i in range(n):
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        y = z[i] - (H @ x_pred)
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        x = x_pred + K @ y
        P = (I - K @ H) @ P_pred

        x_hist[i] = x

    for j, c in enumerate(cols):
        out[f"{c}_est"] = x_hist[:, j]
    return out
