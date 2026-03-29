"""Fault schedule application for the PWR cooling loop digital twin.

Faults are applied as *modifications* to effective parameters/inputs before
evaluating the locked equations (see `gemini.md`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class EffectiveInputs:
    time: float
    heat_input: float
    pump_status: int
    ambient_temp: float
    eta_pump_effective: float
    k_p_effective: float


def _is_active(t: float, start_time: float, end_time: float) -> bool:
    return (t >= start_time) and (t <= end_time)


def apply_faults(
    *,
    base_input: Dict[str, Any],
    params: Any,
    fault_schedule: Optional[List[Dict[str, Any]]],
) -> EffectiveInputs:
    """Apply active faults at time t and return effective values.

    fault_schedule entries must match:
      {"fault_type": str, "start_time": number, "end_time": number, "magnitude": number}

    Semantics (locked):
      - pump_failure: effective_eta = ETA_PUMP * (1 - magnitude), magnitude in [0,1]
      - overheating: Q_in = Q_in * (1 + magnitude)
      - pressure_spike: k_p_effective = K_P * (1 + magnitude)
    """

    t = float(base_input["time"])
    heat_input = float(base_input["heat_input"])
    pump_status = int(base_input["pump_status"])
    ambient_temp = float(base_input["ambient_temp"])

    eta_eff = float(params.ETA_PUMP)
    k_p_eff = float(params.K_P)

    if not fault_schedule:
        return EffectiveInputs(
            time=t,
            heat_input=heat_input,
            pump_status=pump_status,
            ambient_temp=ambient_temp,
            eta_pump_effective=eta_eff,
            k_p_effective=k_p_eff,
        )

    # For pump_failure, use the maximum active magnitude (worst-case) to avoid
    # ambiguous stacking semantics.
    pump_failure_mags: List[float] = []

    for fault in fault_schedule:
        ftype = str(fault.get("fault_type"))
        start_time = float(fault.get("start_time"))
        end_time = float(fault.get("end_time"))
        magnitude = float(fault.get("magnitude", 0.0))

        if not _is_active(t, start_time, end_time):
            continue

        if ftype == "pump_failure":
            pump_failure_mags.append(magnitude)
        elif ftype == "overheating":
            heat_input = heat_input * (1.0 + magnitude)
        elif ftype == "pressure_spike":
            k_p_eff = k_p_eff * (1.0 + magnitude)
        else:
            raise ValueError(f"Unknown fault_type: {ftype}")

    if pump_failure_mags:
        mag = max(pump_failure_mags)
        # Enforce locked magnitude domain.
        if mag < 0.0:
            mag = 0.0
        if mag > 1.0:
            mag = 1.0
        eta_eff = float(params.ETA_PUMP) * (1.0 - mag)

    return EffectiveInputs(
        time=t,
        heat_input=heat_input,
        pump_status=pump_status,
        ambient_temp=ambient_temp,
        eta_pump_effective=eta_eff,
        k_p_effective=k_p_eff,
    )
