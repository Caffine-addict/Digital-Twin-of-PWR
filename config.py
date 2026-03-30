# Locked physics constants (from gemini.md)
SEED = 42
DT = 1

ALPHA = 0.05
K_C = 0.02
K_P = 0.01
ETA_PUMP = 0.9
F_BASE = 50.0

SIGMA = 0.5

# Status thresholds (Isolation Forest decision_function)
WARNING_SCORE_THRESHOLD = -0.1
CRITICAL_SCORE_THRESHOLD = -0.2

# Fault classification thresholds (heuristics, measured signals)
# Nominal flow ~ F_BASE * ETA_PUMP when pump_status=1
FLOW_LOW_THRESHOLD = F_BASE * ETA_PUMP * 0.7
TEMP_HIGH_THRESHOLD = 330.0
PRESSURE_HIGH_THRESHOLD = 170.0

# Maintenance/trend scoring scales
MAINT_FLOW_SLOPE_SCALE = 0.25
MAINT_TEMP_SLOPE_SCALE = 0.10
MAINT_RISK_FLAG_THRESHOLD = 0.65
