"""Reward metrics for adaptive thresholding.

Reward = detection - false_alarm
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def compute_detection_and_false_alarm(df_scored: pd.DataFrame) -> Dict[str, float]:
    if df_scored is None or df_scored.empty:
        return {"detection": 0.0, "false_alarm": 0.0}

    flagged = (df_scored["anomaly_flag"].to_numpy(dtype=int) != 0)

    if "fault_class" in df_scored.columns:
        fault_present = (df_scored["fault_class"].astype(str) != "none").to_numpy(dtype=bool)
    else:
        fault_present = np.zeros(len(df_scored), dtype=bool)

    if fault_present.any():
        detection = float((flagged & fault_present).sum() / fault_present.sum())
    else:
        detection = 0.0

    no_fault = ~fault_present
    if no_fault.any():
        false_alarm = float((flagged & no_fault).sum() / no_fault.sum())
    else:
        false_alarm = 0.0

    return {"detection": detection, "false_alarm": false_alarm}


def compute_reward(df_scored: pd.DataFrame) -> Dict[str, float]:
    m = compute_detection_and_false_alarm(df_scored)
    reward = float(m["detection"] - m["false_alarm"])
    return {**m, "reward": reward}
