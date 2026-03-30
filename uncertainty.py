"""Optional: rolling variance and confidence intervals.

Lightweight:
- numpy + pandas only
Deterministic:
- rolling computations only
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class UncertaintyConfig:
    window: int = 10
    z: float = 1.96


def compute_rolling_uncertainty(
    df: pd.DataFrame,
    *,
    cols: Sequence[str],
    config: UncertaintyConfig = UncertaintyConfig(),
) -> pd.DataFrame:
    w = int(config.window)
    if w < 2:
        raise ValueError("window must be >= 2")

    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        mean = out[c].rolling(w, min_periods=1).mean()
        var = out[c].rolling(w, min_periods=2).var(ddof=0).fillna(0.0)
        std = np.sqrt(var.to_numpy(dtype=float))
        out[f"{c}_var_{w}"] = var
        out[f"{c}_ci_low_{w}"] = mean.to_numpy(dtype=float) - float(config.z) * std
        out[f"{c}_ci_high_{w}"] = mean.to_numpy(dtype=float) + float(config.z) * std
    return out
