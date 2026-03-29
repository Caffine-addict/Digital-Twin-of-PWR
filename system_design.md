# System Design (A.N.T. Architecture)

## A — Acquire / Generate
- Synthetic data generation using the deterministic simulator in `simulate_system.py`.
- Produces time-series for: temperature, pressure (normalized), flow rate.

## N — Normalize / Process
- Fault injection: `inject_faults.py` applies schedule-driven modifications.
- Feature engineering: measured + inputs + engineered features (see `features.py`).
- Scaling: StandardScaler inside the sklearn Pipeline.

## T — Twin / Detect / Visualize
- ML inference: `predict.py` loads the trained Isolation Forest and produces anomaly scores/flags.
- Dashboard: `dashboard.py` renders graphs and a status indicator.

## Module Contracts
- `simulate_system.py`: simulation core; no ML dependencies.
- `inject_faults.py`: fault schedule + modifiers only; no simulation state.
- `train_model.py`: trains model using normal-only data; saves artifact.
- `predict.py`: inference and status mapping.
- `dashboard.py`: Streamlit UI; button-click simulation; stores results in memory.
