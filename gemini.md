# Project Constitution (gemini.md)

## System Overview
This project implements a simulation-lite, real-time digital twin of a simplified PWR cooling loop.

Modeled components:
- Reactor heat source (heat input `Q_in`)
- Coolant loop (lumped temperature state)
- Pump system (flow generation from on/off and efficiency)
- Heat exchanger (captured implicitly via cooling term)
- Sensors: temperature, pressure (normalized), flow rate

Primary goals:
- Deterministic simulation based on simplified physics-like equations
- Fault injection for realistic scenario generation
- Real-time anomaly detection using Isolation Forest
- Streamlit visualization

## Data Schema (STRICT JSON)

### Input
```json
{
  "time": float,
  "heat_input": float,
  "pump_status": int,
  "ambient_temp": float
}
```

### Output
```json
{
  "temperature": float,
  "pressure": float,
  "flow_rate": float,
  "anomaly_flag": int
}
```

## Constants (Locked)
```text
SEED = 42
DT = 1

# Temperature (degC)
T_MIN = 250
T_MAX = 350

# Pressure (normalized units, unitless)
P_MIN = 1
P_MAX = 200

# Flow rate
F_MIN = 0
F_MAX = 100

# Heat input
Q_MIN = 50
Q_MAX = 200

# Model coefficients
ALPHA = 0.05
K_C = 0.02
K_P = 0.01
ETA_PUMP = 0.9
F_BASE = 50

# Noise
SIGMA = 0.5
```

## Mathematical Model (USE GIVEN EQUATIONS ONLY)

### Core State Variables
- Temperature (T)
- Pressure (P)
- Flow Rate (F)

### Governing Equations

1) Temperature Dynamics

T_{t+1} = T_t + \alpha (Q_{in} - k_c F_t (T_t - T_{amb}))

Where:
- Q_{in} = reactor heat input
- k_c = cooling coefficient
- F_t = flow rate
- T_{amb} = ambient temperature

2) Flow Rate (Pump Behavior)

F_t = F_{base} \cdot \eta_{pump} \cdot S_{pump}

Where:
- S_{pump} = 0 or 1 (OFF/ON)
- \eta_{pump} = efficiency

3) Pressure Model

P_t = k_p \cdot F_t \cdot T_t

Where:
- k_p = pressure coefficient

4) Noise Injection (Sensor Realism)

X_{measured} = X_{true} + \mathcal{N}(0, \sigma)

Apply to:
- Temperature
- Pressure
- Flow

## Component Definitions
- ReactorHeatSource: provides `Q_in` each tick from input stream (and fault-adjusted variant).
- PumpSystem: computes flow `F` using pump on/off and effective efficiency.
- CoolantLoop: updates lumped temperature `T` from heat input and cooling term.
- HeatExchanger: represented by the cooling term `k_c * F * (T - T_amb)`.
- Sensors: produce measured `T`, `P`, `F` by applying Gaussian noise after clamping.

## Fault Model (MANDATORY)

Unified schedule schema:
```json
{
  "fault_type": "pump_failure",
  "start_time": 200,
  "end_time": 400,
  "magnitude": 1.0
}
```

Fault types:

1) Pump Failure (efficiency degradation)
- `magnitude in [0,1]`
- Implementation for active fault window:
  - `effective_eta = ETA_PUMP * (1 - magnitude)`
  - Flow equation uses `effective_eta` as \eta_{pump}

2) Overheating
- Implementation for active fault window:
  - `Q_in = Q_in * (1 + magnitude)`

3) Pressure Spike
- Implementation for active fault window:
  - `k_p_effective = K_P * (1 + magnitude)`
  - Pressure equation uses `k_p_effective` as `k_p`

## Noise Model
- Gaussian noise applied to all measured variables
- Single shared sigma: SIGMA = 0.5
- Applied after clamping
- Deterministic via global seed

Applied as:
- T_measured = T_true + N(0, SIGMA)
- P_measured = P_true + N(0, SIGMA)
- F_measured = F_true + N(0, SIGMA)

## Behavioral Rules
- Determinism:
  - All stochastic behavior uses a single global seed: `SEED=42`
  - No uncontrolled randomness
- Fixed timestep: `DT=1`
- Constraints/clamping (applied to true values before noise):
  - `T = clip(T, T_MIN, T_MAX)`
  - `P = clip(P, P_MIN, P_MAX)`
  - `F = clip(F, F_MIN, F_MAX)`
- Tick execution order (STRICT):
  1. Apply faults
  2. Compute flow
  3. Compute temperature
  4. Compute pressure
  5. Clamp
  6. Add noise -> measured values
  7. Store output format
