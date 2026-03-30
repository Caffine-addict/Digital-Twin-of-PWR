"""Scenario analysis for the PWR digital twin.

This module runs predefined scenarios (normal, pump_failure, overheating, pressure_spike, combined_faults)
and computes evaluation metrics for each scenario.

The metrics computed are:
- detection rate
- false positive rate
- detection delay

Results are saved to runs/scenario_results.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from evaluation_metrics import compute_detection_rate, compute_false_positive_rate, compute_detection_delay
from data_sources import load_real_csv
from features import make_features
from inject_faults import apply_faults
from predict import load_model, score_and_flag
from simulate_system import Params, generate_inputs, simulate
from train_model import train_and_save
from config import SEED as GLOBAL_SEED, DT


def _get_or_train_model(model_path: str = "models/isolation_forest.joblib") -> dict:
    """Load existing model or train a new one on normal data."""
    path = Path(model_path)
    if not path.exists():
        # Train on normal data (no faults) as in dashboard.py
        train_and_save(model_path=model_path, n_steps=2000, params=Params())
    return load_model(model_path)


def _run_scenario(
    scenario_name: str,
    fault_schedule: Optional[List[dict]],
    n_steps: int = 1000,
    model_path: str = "models/isolation_forest.joblib",
) -> Dict[str, float]:
    """Run a single scenario and return metrics.

    Parameters
    ----------
    scenario_name : str
        Name of the scenario (for logging).
    fault_schedule : list of dict or None
        Each dict has keys: fault_type, start_time, end_time, magnitude.
        See inject_faults.py for expected format.
    n_steps : int
        Number of time steps to simulate.
    model_path : str
        Path to the trained Isolation Forest model.

    Returns
    -------
    dict
        Dictionary with keys: detection_rate, false_positive_rate, detection_delay.
    """
    # Generate normal input data (no faults in the input, faults are applied via fault_schedule)
    inputs = generate_inputs(n_steps=n_steps, params=Params())

    # Simulate the system with the fault schedule
    df_sim = simulate(
        n_steps=n_steps,
        inputs=inputs,
        fault_schedule=fault_schedule,
        params=Params(),
        seed=GLOBAL_SEED,  # Use global seed for determinism
    )

    # Load or train model
    model_bundle = _get_or_train_model(model_path)

    # Score and flag the simulated data
    df_scored = score_and_flag(df=df_sim, model_bundle=model_bundle)

    # Compute metrics
    dr = compute_detection_rate(df_scored)
    fpr = compute_false_positive_rate(df_scored)
    dd = compute_detection_delay(df_scored)

    return {
        "detection_rate": dr,
        "false_positive_rate": fpr,
        "detection_delay": dd,
    }


def run_scenario_analysis(
    model_path: str = "models/isolation_forest.joblib",
    n_steps: int = 1000,
) -> Dict[str, Dict[str, float]]:
    """Run all predefined scenarios and collect metrics.

    Parameters
    ----------
    model_path : str
        Path to the trained Isolation Forest model.
    n_steps : int
        Number of time steps to simulate for each scenario.

    Returns
    -------
    dict
        Nested dictionary: {scenario_name: {metric_name: value}}
    """
    # Define scenarios as per objective
    scenarios: Dict[str, Optional[List[dict]]] = {
        "normal": None,  # No faults
        "pump_failure": [
            {
                "fault_type": "pump_failure",
                "start_time": 200,
                "end_time": 800,
                "magnitude": 0.5,  # 50% reduction in pump efficiency
            }
        ],
        "overheating": [
            {
                "fault_type": "overheating",
                "start_time": 200,
                "end_time": 800,
                "magnitude": 0.5,  # 50% increase in heat input
            }
        ],
        "pressure_spike": [
            {
                "fault_type": "pressure_spike",
                "start_time": 200,
                "end_time": 800,
                "magnitude": 0.5,  # 50% increase in pressure coefficient
            }
        ],
        "combined_faults": [
            {
                "fault_type": "pump_failure",
                "start_time": 200,
                "end_time": 800,
                "magnitude": 0.3,
            },
            {
                "fault_type": "overheating",
                "start_time": 300,
                "end_time": 900,
                "magnitude": 0.3,
            },
            {
                "fault_type": "pressure_spike",
                "start_time": 400,
                "end_time": 1000,
                "magnitude": 0.3,
            },
        ],
    }

    results: Dict[str, Dict[str, float]] = {}

    for scenario_name, fault_schedule in scenarios.items():
        # Run the scenario
        metrics = _run_scenario(
            scenario_name=scenario_name,
            fault_schedule=fault_schedule,
            n_steps=n_steps,
            model_path=model_path,
        )
        results[scenario_name] = metrics

    # Save results to JSON
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)
    output_path = runs_dir / "scenario_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    # When run directly, execute the scenario analysis and print results
    results = run_scenario_analysis()
    print("Scenario Analysis Results:")
    for scenario, metrics in results.items():
        print(f"\n{scenario}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")