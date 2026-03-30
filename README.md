# Digital Twin of a PWR Cooling Loop (Simulation-Lite)

A deterministic, real-time **digital twin** of a simplified Pressurized Water Reactor (PWR) cooling loop 


- Deterministic seed: `SEED = 42`
- Fixed timestep: `DT = 1`
- Core state variables: Temperature (T), Pressure (P, normalized), Flow Rate (F)
- Fault injection + anomaly detection (Isolation Forest)
- Streamlit dashboard for visualization + predictive risk reporting

## Scope (Simplified)

Modeled components:
- Reactor heat source (`heat_input`)
- Coolant loop (lumped temperature)
- Pump system (flow generation)
- Heat exchanger effect (captured implicitly via cooling term)
- Sensors (T/P/F) with Gaussian noise

This project intentionally avoids complex nuclear simulators, PDEs, and reactor physics libraries.

## Locked Mathematical Model (Equations Only)

State variables: `T`, `P`, `F`

1) Temperature dynamics:

T_{t+1} = T_t + alpha (Q_in - k_c F_t (T_t - T_amb))

2) Pump / flow:

F_t = F_base * eta_pump * S_pump

3) Pressure (normalized):

P_t = k_p * F_t * T_t

4) Sensor noise:

X_measured = X_true + N(0, sigma)

Clamping is applied to true values **before** noise.

## Data Schema (Strict)

Input:
```json
{
  "time": float,
  "heat_input": float,
  "pump_status": int,
  "ambient_temp": float
}
```

Output:
```json
{
  "temperature": float,
  "pressure": float,
  "flow_rate": float,
  "anomaly_flag": int
}
```

## Fault Model

Fault schedule schema:
```json
{
  "fault_type": "pump_failure",
  "start_time": 200,
  "end_time": 400,
  "magnitude": 1.0
}
```

Fault types:
- `pump_failure`: efficiency degradation
  - `effective_eta = ETA_PUMP * (1 - magnitude)` where `magnitude in [0,1]`
- `overheating`: `Q_in = Q_in * (1 + magnitude)`
- `pressure_spike`: `k_p_effective = K_P * (1 + magnitude)`

## Project Layout

- `gemini.md` — project constitution (schema, equations, constraints)
- `simulate_system.py` — deterministic simulator (equations + clamping + noise)
- `inject_faults.py` — fault schedule application
- `features.py` — feature engineering (window=10 + proxies)
- `train_model.py` — trains Isolation Forest on normal-only synthetic data + writes a training report JSON
- `predict.py` — scoring + status mapping
- `dashboard.py` — Streamlit UI (tabs, fault table editor, predictive report)

Docs:
- `system_design.md`, `simulation_logic.md`, `ml_pipeline.md`, `findings.md`, `progress.md`

## Setup

Create a local virtual environment and install dependencies:

```bash
python3 -m venv .venv
./.venv/bin/pip install -U pip
./.venv/bin/pip install numpy pandas scikit-learn streamlit joblib
```

## Run the Dashboard

```bash
./.venv/bin/streamlit run dashboard.py
```

In the UI:
- Click **Retrain model (normal only)** after first setup (shows a training report popup).
- Use the **Fault Schedule** table to add one or more faults.
- Click **Run simulation** to generate a run.
- Click **Run predictive analysis** to simulate forward (default horizon=100) and generate a predictive risk report popup.

## Status Levels

Isolation Forest score thresholds:
- `score < -0.2` -> CRITICAL
- `score < -0.1` -> WARNING
- else -> NORMAL

Dashboard shows:
- Current status: worst over the last 10 steps
- Run status: worst overall during the run

## Determinism Notes

- Simulation noise is deterministic with `SEED = 42`.
- Predictive horizon uses the same "normal" input profile as training.
- Forecast uses a deterministic seed offset for stable repeatability.

## Future Extensions

The system can be extended in several directions:
1. Integration of real-world datasets (if accessible)
2. Use of advanced models such as LSTM for time-series prediction
3. Expansion to full reactor digital twin
4. Adaptive thresholding using reinforcement learning

## License

No license file is included yet. Add one if you plan to distribute this publicly.
