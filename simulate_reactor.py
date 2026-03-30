"""Optional expanded reactor simulator.

Adds ONLY:
- Steam generator (heat exchange term)
- Turbine output (derived)

Does NOT modify simulate_system.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from inject_faults import apply_faults


@dataclass(frozen=True)
class ParamsReactor:
    # Determinism / time
    SEED: int = 42
    DT: int = 1

    # Ranges
    T_MIN: float = 250.0
    T_MAX: float = 350.0
    P_MIN: float = 1.0
    P_MAX: float = 200.0
    F_MIN: float = 0.0
    F_MAX: float = 100.0
    Q_MIN: float = 50.0
    Q_MAX: float = 200.0

    # Coefficients (primary loop)
    ALPHA: float = 0.05
    K_C: float = 0.02
    K_P: float = 0.01
    ETA_PUMP: float = 0.9
    F_BASE: float = 50.0

    # Steam generator coupling
    K_SG: float = 0.01
    ALPHA_S: float = 0.02
    K_S_COOL: float = 0.005
    TURBINE_EFF: float = 0.35

    # Noise
    SIGMA: float = 0.5


def _clip(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


def simulate_reactor(
    *,
    n_steps: int,
    inputs: List[Dict[str, float]],
    fault_schedule: Optional[List[Dict[str, Any]]] = None,
    params: ParamsReactor = ParamsReactor(),
    seed: Optional[int] = None,
    initial_temperature: float = 300.0,
    initial_secondary_temperature: float = 290.0,
) -> pd.DataFrame:
    if len(inputs) < int(n_steps):
        raise ValueError("inputs length must be >= n_steps")

    rng = np.random.default_rng(params.SEED if seed is None else int(seed))

    T_p = _clip(float(initial_temperature), params.T_MIN, params.T_MAX)
    T_s = float(initial_secondary_temperature)
    F = 0.0
    P = _clip(float(params.K_P) * float(params.F_BASE) * T_p, params.P_MIN, params.P_MAX)

    rows: List[Dict[str, Any]] = []
    for i in range(int(n_steps)):
        base_inp = inputs[i]
        eff = apply_faults(base_input=base_inp, params=params, fault_schedule=fault_schedule)

        F = float(params.F_BASE) * float(eff.eta_pump_effective) * int(eff.pump_status)
        Q_in = _clip(float(eff.heat_input), params.Q_MIN, params.Q_MAX)

        sg_transfer = float(params.K_SG) * F * (T_p - float(eff.ambient_temp))

        T_next = T_p + float(params.ALPHA) * (
            Q_in - float(params.K_C) * F * (T_p - float(eff.ambient_temp)) - sg_transfer
        )

        T_s_next = T_s + float(params.ALPHA_S) * (
            sg_transfer - float(params.K_S_COOL) * (T_s - float(eff.ambient_temp))
        )

        P_next = float(eff.k_p_effective) * F * T_next

        T_true = _clip(T_next, params.T_MIN, params.T_MAX)
        F_true = _clip(F, params.F_MIN, params.F_MAX)
        P_true = _clip(P_next, params.P_MIN, params.P_MAX)

        T_meas = float(T_true + rng.normal(0.0, float(params.SIGMA)))
        P_meas = float(P_true + rng.normal(0.0, float(params.SIGMA)))
        F_meas = float(F_true + rng.normal(0.0, float(params.SIGMA)))

        turbine_output = float(params.TURBINE_EFF) * max(0.0, sg_transfer)

        rows.append(
            {
                "time": float(eff.time),
                "heat_input": float(base_inp["heat_input"]),
                "pump_status": int(base_inp["pump_status"]),
                "ambient_temp": float(base_inp["ambient_temp"]),
                "heat_input_effective": float(Q_in),
                "eta_pump_effective": float(eff.eta_pump_effective),
                "k_p_effective": float(eff.k_p_effective),
                "temperature_true": float(T_true),
                "pressure_true": float(P_true),
                "flow_rate_true": float(F_true),
                "temperature": float(T_meas),
                "pressure": float(P_meas),
                "flow_rate": float(F_meas),
                "steam_generator_transfer": float(sg_transfer),
                "secondary_temperature": float(T_s_next),
                "turbine_output": float(turbine_output),
                "anomaly_flag": 0,
            }
        )

        T_p, P, T_s = T_true, P_true, float(T_s_next)

    return pd.DataFrame(rows)
