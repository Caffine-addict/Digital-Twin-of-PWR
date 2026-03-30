"""Error analysis for the PWR digital twin.

This module analyzes the results of anomaly detection to identify:
- False positives
- Missed anomalies (false negatives)
- Common failure patterns

It returns a dictionary with counts and a list of common failure patterns.
"""
from __future__ import annotations

from typing import Dict, List
import pandas as pd


def analyze_errors(df: pd.DataFrame) -> Dict[str, object]:
    """Analyze errors in anomaly detection.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns:
        - 'anomaly_flag': 0 (normal) or 1 (anomaly)
        - 'fault_class': string, 'none' for normal operation, otherwise fault type
        - 'temperature', 'pressure', 'flow_rate': sensor values

    Returns
    -------
    dict
        Keys:
        - 'false_positives': int
        - 'missed_anomalies': int
        - 'common_failure_patterns': list of strings
    """
    if df is None or df.empty:
        return {
            "false_positives": 0,
            "missed_anomalies": 0,
            "common_failure_patterns": [],
        }

    # False positives: anomaly_flag == 1 but fault_class == 'none'
    false_positives = df[(df['anomaly_flag'] == 1) & (df['fault_class'] == 'none')]
    fp_count = len(false_positives)

    # Missed anomalies: anomaly_flag == 0 but fault_class != 'none'
    missed_anomalies = df[(df['anomaly_flag'] == 0) & (df['fault_class'] != 'none')]
    missed_count = len(missed_anomalies)

    # Determine common failure patterns
    patterns: List[str] = []

    # Pattern 1: High temperature fluctuation -> false positive
    if fp_count > 0:
        # Compute temperature standard deviation in false positive instances
        temp_std_fp = false_positives['temperature'].std()
        # If temperature fluctuation is high (above some threshold), consider it a pattern
        # Threshold chosen arbitrarily for demonstration; in practice, this would be tuned.
        if temp_std_fp > 1.0:  # e.g., more than 1 degree std dev
            patterns.append("high_temp_fluctuation -> false positive")

    # Pattern 2: Slow drift -> missed detection
    if missed_count > 0:
        # Compute the trend (difference between end and start) for each sensor in missed anomalies
        # We'll approximate by taking the mean of the sensor values and see if they are biased.
        # For simplicity, we check if the mean temperature is above normal (indicating overheating trend)
        # or mean flow is below normal (indicating pump failure trend).
        temp_mean = missed_anomalies['temperature'].mean()
        flow_mean = missed_anomalies['flow_rate'].mean()
        pressure_mean = missed_anomalies['pressure'].mean()

        # Define normal ranges (these could be loaded from config or computed from normal data)
        # For demonstration, we use arbitrary thresholds.
        if temp_mean > 305.0:  # Example threshold for overheating
            patterns.append("slow_temp_drift -> missed detection")
        if flow_mean < 45.0:  # Example threshold for low flow
            patterns.append("slow_flow_drift -> missed detection")
        if pressure_mean > 180.0:  # Example threshold for high pressure
            patterns.append("slow_pressure_drift -> missed detection")

    # Remove duplicates while preserving order
    seen = set()
    unique_patterns = []
    for p in patterns:
        if p not in seen:
            seen.add(p)
            unique_patterns.append(p)

    return {
        "false_positives": int(fp_count),
        "missed_anomalies": int(missed_count),
        "common_failure_patterns": unique_patterns,
    }