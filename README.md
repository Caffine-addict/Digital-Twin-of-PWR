# Digital Twin of a PWR Cooling Loop

A deterministic digital twin of a simplified pressurized-water-reactor (PWR) cooling loop. It
simulates temperature, pressure and coolant flow, injects faults, and flags anomalies with an
Isolation Forest, all driven from a Streamlit dashboard. An evaluation harness then scores the
detector on simulated fault scenarios and on real accident data.

## What it does

- **Simulator** (`simulate_system.py`): lumped heat-balance model of the loop with a pump, a heat
  source and noisy temperature, pressure and flow sensors. Fixed seed and timestep, so every run
  is reproducible.
- **Fault injection** (`inject_faults.py`): pump degradation, overheating and pressure spikes,
  scheduled by start time, end time and magnitude.
- **Anomaly detection** (`features.py`, `train_model.py`, `predict.py`): sliding-window features
  feed an Isolation Forest trained on normal operation only, with NORMAL / WARNING / CRITICAL
  thresholds.
- **Forecasting** (`lstm_predictor.py`, `predict_future.py`): an optional LSTM, used when PyTorch is
  installed, projects the signals forward for a predictive risk report.
- **Adaptive thresholds** (`threshold_agent.py`): an epsilon-greedy bandit that tunes the alert
  thresholds from feedback.
- **Evaluation** (`evaluation_metrics.py`, `scenario_analysis.py`, `error_analysis.py`,
  `ablation.py`): detection rate, false-positive rate and detection delay per fault scenario,
  plus error analysis on the real [NPPAD](https://springernature.figshare.com/articles/dataset/NPPAD_An_Open_Time-series_Dataset_Covering_Various_Accidents_for_Nuclear_Power_Plants/21299880) accident
  dataset (`nppad_parser.py` converts its MDB files).

## Model

State variables: temperature `T`, normalized pressure `P`, flow rate `F`.

```
T(t+1) = T(t) + alpha * (Q_in - k_c * F(t) * (T(t) - T_amb))
F(t)   = F_base * eta_pump * S_pump
P(t)   = k_p * F(t) * T(t)
X_measured = X_true + N(0, sigma)        # true values clamped before noise
```

Faults: `pump_failure` scales pump efficiency by `(1 - magnitude)`, `overheating` scales heat input
by `(1 + magnitude)`, and `pressure_spike` scales `k_p` by `(1 + magnitude)`.

## Results

| Scenario | Detection rate | False-positive rate |
|---|---|---|
| Pump failure | 78.4% | 4.0% |
| Combined faults | 41.8% | 6.5% |
| Overheating | 0% | 3.3% |
| Pressure spike | 0% | 3.3% |

Pump failures are caught well. Overheating and pressure spikes are not detected yet, which points to
fault-specific features as the next step. On real NPPAD
accident data the detector missed nothing but raised 5,791 false alarms, so a model trained only on
simulated data does not transfer to a real plant without retraining.

## Run it

```bash
python3 -m venv .venv
./.venv/bin/pip install numpy pandas scikit-learn streamlit joblib altair
./.venv/bin/streamlit run dashboard.py
```

In the dashboard: **Retrain model (normal only)** once, add faults in the **Fault Schedule** table,
**Run simulation**, then **Run predictive analysis** for a forward-looking risk report.

Design notes live in `system_design.md`, `simulation_logic.md`, `ml_pipeline.md` and
`MASTER_DOCUMENTATION.md`.
