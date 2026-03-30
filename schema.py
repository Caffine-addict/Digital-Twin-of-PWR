"""Schema helpers for real-world data ingestion.

Contract (real CSV): required columns
  - time, temperature, pressure, flow_rate
"""

from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = ["time", "temperature", "pressure", "flow_rate"]


def validate_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"real data missing required columns: {missing}")


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in REQUIRED_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)
    return out


def add_missing_inputs(
    df: pd.DataFrame,
    *,
    heat_input: float,
    ambient_temp: float,
    pump_status: int,
) -> pd.DataFrame:
    """Add required context columns for downstream feature engineering."""
    out = df.copy()
    out["heat_input"] = float(heat_input)
    out["heat_input_effective"] = float(heat_input)
    out["ambient_temp"] = float(ambient_temp)
    out["pump_status"] = int(pump_status)

    # Placeholders that the simulator normally provides.
    out["eta_pump_effective"] = float("nan")
    out["k_p_effective"] = float("nan")
    out["temperature_true"] = float("nan")
    out["pressure_true"] = float("nan")
    out["flow_rate_true"] = float("nan")
    out["anomaly_flag"] = 0
    return out
