"""Results table comparing anomaly detection methods.

This module creates a comparison table of different anomaly detection methods:
- Isolation Forest (existing)
- Z-score (rule-based)
- MAD (robust rule-based)

It computes detection rate, false positive rate, and detection delay for each method.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import os
from pathlib import Path

from evaluation_metrics import (
    compute_false_positive_rate,
    compute_detection_rate,
    compute_detection_delay,
)
from model_benchmark import benchmark
from predict import score_and_flag
from features import make_features


def _get_method_flags(
    df: pd.DataFrame, model_bundle: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get anomaly flags for all three methods.

    Returns
    -------
    Tuple of (isolation_forest_flags, zscore_flags, mad_flags)
    each as boolean numpy arrays.
    """
    # Baseline (Isolation Forest)
    base = score_and_flag(df, model_bundle)
    base_flag = (base["anomaly_flag"].to_numpy(dtype=int) != 0)

    # Rule methods
    feat = make_features(df=df, window=10, eps=1e-6)
    X = feat[["temperature", "pressure", "flow_rate"]].to_numpy(dtype=float)

    # Z-score method (threshold = 3.0)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd == 0.0, 1.0, sd)
    z = (X - mu) / sd
    z_flag = (np.abs(z) > 3.0).any(axis=1)

    # MAD method (threshold = 3.5)
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    mad = np.where(mad == 0.0, 1.0, mad)
    rz = 0.6745 * (X - med) / mad
    mad_flag = (np.abs(rz) > 3.5).any(axis=1)

    return base_flag, z_flag, mad_flag


def create_results_table(
    df: pd.DataFrame, model_bundle: Dict[str, Any]
) -> pd.DataFrame:
    """Create a comparison table of anomaly detection methods.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with sensor data and fault_class column.
    model_bundle : dict
        Trained model bundle from predict.load_model()

    Returns
    -------
    pd.DataFrame
        Comparison table with columns:
        - method: method name
        - detection_rate: fraction of faults correctly detected
        - false_positive_rate: fraction of normal samples flagged as anomalous
        - detection_delay: average time steps to detect faults
    """
    if df is None or df.empty:
        # Return empty table with expected columns
        return pd.DataFrame(
            columns=["method", "detection_rate", "false_positive_rate", "detection_delay"]
        )

    # Get flags for each method
    forest_flag, zscore_flag, mad_flag = _get_method_flags(df, model_bundle)

    # Create dataframes with method flags for metric computation
    methods = [
        ("Isolation Forest", forest_flag),
        ("Z-Score", zscore_flag),
        ("MAD", mad_flag),
    ]

    rows = []
    for method_name, flag_array in methods:
        # Create a copy of df with the method's anomaly flag
        df_method = df.copy()
        df_method["anomaly_flag"] = flag_array.astype(int)

        # Compute metrics
        fpr = compute_false_positive_rate(df_method)
        dr = compute_detection_rate(df_method)
        dd = compute_detection_delay(df_method)

        rows.append(
            {
                "method": method_name,
                "detection_rate": dr,
                "false_positive_rate": fpr,
                "detection_delay": dd,
            }
        )

    results_df = pd.DataFrame(rows)

    # Save to CSV
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)
    output_path = runs_dir / "results_table.csv"
    results_df.to_csv(output_path, index=False)

    return results_df


def create_results_table_from_benchmark(
    df: pd.DataFrame, model_bundle: Dict[str, Any]
) -> pd.DataFrame:
    """Alternative implementation using the existing benchmark function.

    This function demonstrates how to extract the needed information
    from the existing benchmark function if we wanted to reuse it more.
    However, the benchmark function doesn't return the actual flags,
    so we reimplement the flag extraction here for clarity.
    """
    # For now, we'll use the direct implementation above
    return create_results_table(df, model_bundle)