"""Predefined experiments/scenarios for the digital twin.

Lightweight:
- uses existing simulator + predictor
- deterministic scenarios
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

import config as cfg
from simulate_system import Params, generate_inputs, simulate
from train_model import train_and_save
from predict import load_model, score_and_flag
from evaluation_metrics import compute_all_metrics, compute_prediction_mae
from predict_next import predict_next


@dataclass(frozen=True)
class Scenario:
    name: str
    n_steps: int
    fault_schedule: Optional[List[Dict[str, Any]]]
    heat_input: float = 120.0
    ambient_temp: float = 290.0
    pump_status: int = 1


def predefined_scenarios() -> List[Scenario]:
    return [
        Scenario(name="normal", n_steps=800, fault_schedule=None),
        Scenario(
            name="pump_failure",
            n_steps=800,
            fault_schedule=[{"fault_type": "pump_failure", "start_time": 200, "end_time": 400, "magnitude": 1.0}],
        ),
        Scenario(
            name="overheating",
            n_steps=800,
            fault_schedule=[{"fault_type": "overheating", "start_time": 200, "end_time": 400, "magnitude": 1.0}],
        ),
        Scenario(
            name="pressure_spike",
            n_steps=800,
            fault_schedule=[{"fault_type": "pressure_spike", "start_time": 200, "end_time": 400, "magnitude": 1.0}],
        ),
        Scenario(
            name="combo_heat_pump",
            n_steps=800,
            fault_schedule=[
                {"fault_type": "overheating", "start_time": 200, "end_time": 400, "magnitude": 1.0},
                {"fault_type": "pump_failure", "start_time": 250, "end_time": 450, "magnitude": 1.0},
            ],
        ),
    ]


def _prediction_mae_for_run(
    df_scored: pd.DataFrame,
    *,
    predictor_method: str,
    lstm_state: Any,
) -> Dict[str, Any]:
    if df_scored is None or len(df_scored) < 25:
        return {"prediction_mae": None}

    # Evaluate one-step window->next10 at a fixed anchor for determinism.
    anchor = 20
    window = df_scored.iloc[anchor - 10 : anchor]
    future_true = df_scored.iloc[anchor : anchor + 10]
    pred = predict_next(window, method=str(predictor_method), lstm_state=lstm_state, horizon=10)
    return {
        "prediction_mae": {
            "T": compute_prediction_mae(y_true=future_true["temperature"].to_numpy(dtype=float), y_pred=pred["T_future"]),
            "P": compute_prediction_mae(y_true=future_true["pressure"].to_numpy(dtype=float), y_pred=pred["P_future"]),
            "F": compute_prediction_mae(y_true=future_true["flow_rate"].to_numpy(dtype=float), y_pred=pred["F_future"]),
        }
    }


def run_scenario(
    scenario: Scenario,
    *,
    model_path: str = "models/isolation_forest.joblib",
    predictor_method: str = "linear",
    lstm_state: Any = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    params = Params()
    if not pd.isna(model_path):
        train_and_save(model_path=model_path, n_steps=2000, params=params)
    model_bundle = load_model(model_path)

    inputs = generate_inputs(
        n_steps=int(scenario.n_steps),
        heat_input=float(scenario.heat_input),
        pump_status=int(scenario.pump_status),
        ambient_temp=float(scenario.ambient_temp),
        dt=float(params.DT),
        profile="normal",
    )
    df = simulate(n_steps=int(scenario.n_steps), inputs=inputs, fault_schedule=scenario.fault_schedule, params=params)
    df_scored = score_and_flag(df, model_bundle, thresholds=thresholds)

    metrics = compute_all_metrics(df_scored=df_scored, fault_schedule=scenario.fault_schedule)
    pred_metrics = _prediction_mae_for_run(df_scored, predictor_method=predictor_method, lstm_state=lstm_state)

    return {
        "scenario": {
            "name": scenario.name,
            "n_steps": int(scenario.n_steps),
            "fault_schedule": scenario.fault_schedule,
        },
        "metrics": {**metrics, **pred_metrics},
    }


def run_all(
    *,
    model_path: str = "models/isolation_forest.joblib",
    predictor_method: str = "linear",
    lstm_state: Any = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    results = []
    for s in predefined_scenarios():
        results.append(run_scenario(s, model_path=model_path, predictor_method=predictor_method, lstm_state=lstm_state, thresholds=thresholds))
    return {"results": results}
