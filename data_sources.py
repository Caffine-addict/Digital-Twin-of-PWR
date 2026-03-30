"""Data sources for the PWR digital twin platform.

Real-world data contract:
- CSV path: ./data/real_data.csv
- Required columns: time, temperature, pressure, flow_rate
- Missing inputs filled with defaults:
    heat_input=100, ambient_temp=300, pump_status=1
- DT=1 enforced
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import schema
import resample


REAL_DATA_DEFAULT_PATH = Path("data/real_data.csv")


def load_real_csv(
    path: Path = REAL_DATA_DEFAULT_PATH,
    *,
    heat_input: float = 100.0,
    ambient_temp: float = 300.0,
    pump_status: int = 1,
) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"real data CSV not found: {p}")

    df = pd.read_csv(p)
    schema.validate_required_columns(df)
    df = schema.coerce_numeric(df)
    df = resample.enforce_dt1(df)
    df = schema.add_missing_inputs(
        df,
        heat_input=float(heat_input),
        ambient_temp=float(ambient_temp),
        pump_status=int(pump_status),
    )
    return df
