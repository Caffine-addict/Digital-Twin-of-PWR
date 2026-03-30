"""Resampling utilities.

Requirement: DT = 1 enforced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def enforce_dt1(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new df with time strictly on a dt=1 grid.

    - sorts by time
    - creates a dt=1 grid from ceil(min(time)) to floor(max(time))
    - interpolates temperature/pressure/flow_rate onto that grid
    """
    out = df.copy()
    out = out.sort_values("time").reset_index(drop=True)

    t = out["time"].to_numpy(dtype=float)
    if t.size == 0:
        raise ValueError("empty time series")

    t0 = float(np.ceil(t.min()))
    t1 = float(np.floor(t.max()))
    if t1 < t0:
        raise ValueError("invalid time bounds for dt=1 enforcement")

    grid = np.arange(t0, t1 + 1.0, 1.0, dtype=float)

    def interp(col: str) -> np.ndarray:
        y = out[col].to_numpy(dtype=float)
        return np.interp(grid, t, y)

    res = pd.DataFrame(
        {
            "time": grid,
            "temperature": interp("temperature"),
            "pressure": interp("pressure"),
            "flow_rate": interp("flow_rate"),
        }
    )
    return res
