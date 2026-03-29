"""Feature engineering for the PWR cooling loop digital twin.

These features are derived from measured signals and known inputs only.
They do not modify the locked physics equations.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


FEATURE_ORDER: List[str] = [
    "temperature",
    "pressure",
    "flow_rate",
    "heat_input_effective",
    "ambient_temp",
    "pump_status",
    "dT",
    "dP",
    "dF",
    "ddT",
    "ddP",
    "T_minus_amb",
    "cooling_proxy",
    "k_p_hat",
    "T_mean_10",
    "T_std_10",
    "P_mean_10",
    "P_std_10",
    "F_mean_10",
    "F_std_10",
]


def make_features(*, df: pd.DataFrame, window: int = 10, eps: float = 1e-6) -> pd.DataFrame:
    """Return a copy of df with engineered features added.

    Required columns:
      - temperature, pressure, flow_rate
      - heat_input_effective, ambient_temp, pump_status
    """

    out = df.copy()

    # First derivatives
    out["dT"] = out["temperature"].diff().fillna(0.0)
    out["dP"] = out["pressure"].diff().fillna(0.0)
    out["dF"] = out["flow_rate"].diff().fillna(0.0)

    # Second differences
    out["ddT"] = out["dT"].diff().fillna(0.0)
    out["ddP"] = out["dP"].diff().fillna(0.0)

    # Proxies aligned with locked equation structure.
    out["T_minus_amb"] = out["temperature"] - out["ambient_temp"]
    out["cooling_proxy"] = out["flow_rate"] * out["T_minus_amb"]

    denom = (out["flow_rate"] * out["temperature"]).to_numpy(dtype=float)
    denom = np.where(np.abs(denom) < float(eps), float(eps), denom)
    out["k_p_hat"] = out["pressure"].to_numpy(dtype=float) / denom

    # Rolling stats (causal)
    w = int(window)
    out["T_mean_10"] = out["temperature"].rolling(w, min_periods=1).mean()
    out["T_std_10"] = out["temperature"].rolling(w, min_periods=1).std(ddof=0)
    out["P_mean_10"] = out["pressure"].rolling(w, min_periods=1).mean()
    out["P_std_10"] = out["pressure"].rolling(w, min_periods=1).std(ddof=0)
    out["F_mean_10"] = out["flow_rate"].rolling(w, min_periods=1).mean()
    out["F_std_10"] = out["flow_rate"].rolling(w, min_periods=1).std(ddof=0)

    # Safety: std can still contain NaN if upstream data is NaN.
    for c in ("T_std_10", "P_std_10", "F_std_10"):
        out[c] = out[c].fillna(0.0)

    return out
