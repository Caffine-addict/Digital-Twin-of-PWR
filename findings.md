# Findings (Locked Decisions)

## Modeling
- Pressure is normalized and unitless; range: `P_MIN=1` to `P_MAX=200`
- Time step is fixed: `DT = 1`
- Deterministic seed: `SEED = 42`

## Constants
- Ranges: `T_MIN=250`, `T_MAX=350`; `F_MIN=0`, `F_MAX=100`; `Q_MIN=50`, `Q_MAX=200`
- Coefficients: `ALPHA=0.05`, `K_C=0.02`, `K_P=0.01`, `ETA_PUMP=0.9`, `F_BASE=50`
- Noise: shared `SIGMA=0.5` for T/P/F

## Strict Tick Order
1) Apply faults 2) Compute flow 3) Compute temperature 4) Compute pressure 5) Clamp 6) Add noise 7) Store

## Fault Magnitude Semantics
- Pump failure: `magnitude in [0,1]` degrades pump efficiency
  - `effective_eta = ETA_PUMP * (1 - magnitude)`
  - `F_t = F_BASE * effective_eta * pump_status`
- Overheating: `Q_in = Q_in * (1 + magnitude)`
- Pressure spike: `k_p_effective = K_P * (1 + magnitude)`

## ML Pipeline
- Train on normal-only synthetic data
- Features: measured + inputs + engineered (window=10)
  - Base: `temperature`, `pressure`, `flow_rate`
  - Inputs: `heat_input_effective`, `ambient_temp`, `pump_status`
  - Derived: `dT`, `dP`, `dF`, `ddT`, `ddP`, `T_minus_amb`, `cooling_proxy`, `k_p_hat`, rolling mean/std for T/P/F
- Status thresholds (Isolation Forest score):
  - `score < -0.2` -> CRITICAL
  - `score < -0.1` -> WARNING
  - else -> NORMAL
