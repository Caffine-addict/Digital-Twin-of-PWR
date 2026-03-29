"""Deterministic simulation-lite digital twin for a simplified PWR cooling loop.

Implements ONLY the locked equations from `gemini.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from inject_faults import apply_faults


@dataclass(frozen=True)
class Params:
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

    # Coefficients
    ALPHA: float = 0.05
    K_C: float = 0.02
    K_P: float = 0.01
    ETA_PUMP: float = 0.9
    F_BASE: float = 50.0

    # Noise
    SIGMA: float = 0.5


def _clip(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


def generate_inputs(
    *,
    n_steps: int,
    heat_input: float = 120.0,
    pump_status: int = 1,
    ambient_temp: float = 290.0,
    start_time: float = 0.0,
    dt: float = 1.0,
    profile: str = "normal",
) -> List[Dict[str, float]]:
    """Generate deterministic synthetic inputs matching the strict schema."""

    inputs: List[Dict[str, float]] = []
    for i in range(int(n_steps)):
        t = float(start_time + i * dt)

        q = float(heat_input)
        ta = float(ambient_temp)

        # Deterministic variability (no RNG) to give the ML model a meaningful
        # normal operating envelope.
        if profile == "normal":
            q = q + 5.0 * float(np.sin(2.0 * np.pi * i / 200.0))
            ta = ta + 1.0 * float(np.sin(2.0 * np.pi * i / 500.0))
        elif profile == "steady":
            pass
        else:
            raise ValueError(f"Unknown profile: {profile}")

        inputs.append(
            {
                "time": t,
                "heat_input": q,
                "pump_status": int(pump_status),
                "ambient_temp": ta,
            }
        )
    return inputs


def simulate(
    *,
    n_steps: int,
    inputs: Optional[List[Dict[str, float]]] = None,
    fault_schedule: Optional[List[Dict[str, Any]]] = None,
    params: Params = Params(),
    seed: Optional[int] = None,
    initial_temperature: float = 300.0,
) -> pd.DataFrame:
    """Run the digital twin and return a time-series DataFrame.

    Output columns include strict schema names:
      - temperature, pressure, flow_rate, anomaly_flag
    Additional columns are included for debugging/training (e.g. *_true).
    """

    if inputs is None:
        inputs = generate_inputs(n_steps=n_steps, dt=float(params.DT))
    if len(inputs) < int(n_steps):
        raise ValueError("inputs length must be >= n_steps")

    rng = np.random.default_rng(params.SEED if seed is None else int(seed))

    # Initialize state.
    T = _clip(float(initial_temperature), params.T_MIN, params.T_MAX)
    F = 0.0
    P = _clip(float(params.K_P) * float(params.F_BASE) * T, params.P_MIN, params.P_MAX)

    rows: List[Dict[str, Any]] = []

    for i in range(int(n_steps)):
        base_inp = inputs[i]

        eff = apply_faults(base_input=base_inp, params=params, fault_schedule=fault_schedule)

        # 2) Compute flow (locked pump equation).
        F = float(params.F_BASE) * float(eff.eta_pump_effective) * int(eff.pump_status)

        # 3) Compute temperature (locked temperature equation).
        # Clamp heat input to its defined domain as an input constraint.
        Q_in = _clip(float(eff.heat_input), params.Q_MIN, params.Q_MAX)
        T_next = T + float(params.ALPHA) * (
            Q_in - float(params.K_C) * F * (T - float(eff.ambient_temp))
        )

        # 4) Compute pressure (locked pressure equation).
        P_next = float(eff.k_p_effective) * F * T_next

        # 5) Clamp true values.
        T_true = _clip(T_next, params.T_MIN, params.T_MAX)
        F_true = _clip(F, params.F_MIN, params.F_MAX)
        P_true = _clip(P_next, params.P_MIN, params.P_MAX)

        # 6) Add noise -> measured values.
        T_meas = float(T_true + rng.normal(0.0, float(params.SIGMA)))
        P_meas = float(P_true + rng.normal(0.0, float(params.SIGMA)))
        F_meas = float(F_true + rng.normal(0.0, float(params.SIGMA)))

        # 7) Store (strict schema field names + useful extras).
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
                "anomaly_flag": 0,
            }
        )

        T, P = T_true, P_true

    return pd.DataFrame(rows)
