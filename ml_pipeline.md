# ML Pipeline

## Training (Normal Data Only)
Pipeline:
- `simulate_system.py` -> generate normal-only synthetic time-series
- `train_model.py` -> train Isolation Forest model and save artifact

Features (per timestep) (measured + inputs + engineered, window=10):
- Base measured: `temperature`, `pressure`, `flow_rate`
- Inputs/context: `heat_input_effective`, `ambient_temp`, `pump_status`
- Derivatives: `dT`, `dP`, `dF`, `ddT`, `ddP`
- Proxies: `T_minus_amb`, `cooling_proxy`, `k_p_hat`
- Rolling stats: `T_mean_10`, `T_std_10`, `P_mean_10`, `P_std_10`, `F_mean_10`, `F_std_10`

Model:
- `StandardScaler` -> `IsolationForest`
- Deterministic via `random_state=SEED`

## Testing / Runtime
Pipeline:
- `simulate_system.py` -> base simulation
- `inject_faults.py` -> apply fault schedule
- `predict.py` -> compute anomaly score and map to status

Status mapping (Isolation Forest score):
```python
if score < -0.2:
    status = "CRITICAL"
elif score < -0.1:
    status = "WARNING"
else:
    status = "NORMAL"
```
