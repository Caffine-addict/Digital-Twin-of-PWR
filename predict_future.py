from __future__ import annotations

from typing import Any, Dict, List
import numpy as np

WINDOW = 10
HORIZON = 10
DT = 1.0  # explicit dt=1 as required


def _as_series(window: Any, key: str) -> np.ndarray:
    # Accept pd.DataFrame-like, list[dict], or dict[str, list]
    if hasattr(window, "tail") and hasattr(window, "columns") and key in window.columns:  # DataFrame-like
        vals = window[key].tail(WINDOW).to_numpy(dtype=float)
        return vals
    if isinstance(window, dict):
        return np.asarray(window[key][-WINDOW:], dtype=float)
    if isinstance(window, list):
        return np.asarray([float(row[key]) for row in window[-WINDOW:]], dtype=float)
    raise TypeError("data_window must be a DataFrame-like, list[dict], or dict of lists")


def _linreg_forecast(y: np.ndarray) -> List[float]:
    # Fit y = a*t + b using least squares on t=0..9 (dt=1), predict t=10..19
    t = np.arange(WINDOW, dtype=float) * DT
    A = np.column_stack([t, np.ones_like(t)])
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    t_future = np.arange(WINDOW, WINDOW + HORIZON, dtype=float) * DT
    y_future = a * t_future + b
    return [float(v) for v in y_future.tolist()]


def predict_future(data_window: Any) -> Dict[str, List[float]]:
    """
    Input: last ~10 points window
    Output:
      { "T_future": [...10...], "P_future": [...10...], "F_future": [...10...] }
    """
    T = _as_series(data_window, "temperature")
    P = _as_series(data_window, "pressure")
    F = _as_series(data_window, "flow_rate")

    if len(T) < WINDOW or len(P) < WINDOW or len(F) < WINDOW:
        raise ValueError("data_window must contain at least 10 points for temperature/pressure/flow_rate")

    return {
        "T_future": _linreg_forecast(T),
        "P_future": _linreg_forecast(P),
        "F_future": _linreg_forecast(F),
    }
