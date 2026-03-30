"""Train Isolation Forest on normal-only synthetic data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config as cfg
from simulate_system import Params, generate_inputs, simulate

from features import FEATURE_ORDER, make_features


def _quantiles(values, qs):
    return {f"p{int(q * 100)}": float(pd.Series(values).quantile(q)) for q in qs}


def train_and_save(
    *,
    model_path: str,
    n_steps: int = 2000,
    params: Params = Params(),
    seed: Optional[int] = None,
) -> str:
    inputs = generate_inputs(n_steps=n_steps, dt=float(params.DT), profile="normal")
    df = simulate(n_steps=n_steps, inputs=inputs, fault_schedule=None, params=params, seed=seed)
    df = make_features(df=df, window=10, eps=1e-6)

    X = df[FEATURE_ORDER].to_numpy(dtype=float)

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "iso_forest",
                IsolationForest(
                    n_estimators=200,
                    contamination="auto",
                    random_state=params.SEED if seed is None else int(seed),
                ),
            ),
        ]
    )
    pipe.fit(X)

    scores = pipe.decision_function(X)
    report = {
        "seed": int(params.SEED if seed is None else seed),
        "n_steps": int(n_steps),
        "feature_order": list(FEATURE_ORDER),
        "model": {
            "type": "IsolationForest",
            "n_estimators": 200,
            "contamination": "auto",
        },
        "score_quantiles": _quantiles(scores, [0.01, 0.05, 0.50, 0.95, 0.99]),
        "expected_rates_on_normal": {
            "warning_rate": float((scores < float(cfg.WARNING_SCORE_THRESHOLD)).mean()),
            "critical_rate": float((scores < float(cfg.CRITICAL_SCORE_THRESHOLD)).mean()),
        },
    }

    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": pipe,
            "feature_order": list(FEATURE_ORDER),
            "seed": params.SEED if seed is None else int(seed),
        },
        path,
    )

    report_path = path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="models/isolation_forest.joblib")
    ap.add_argument("--n-steps", type=int, default=2000)
    args = ap.parse_args()

    saved = train_and_save(model_path=args.model_path, n_steps=args.n_steps)
    print(saved)


if __name__ == "__main__":
    main()
