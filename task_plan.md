# PWR Cooling Loop Digital Twin (B.L.A.S.T. / A.N.T.)

## Phase 0 — Initialization (Strict Gate)
- Create project memory docs: `task_plan.md`, `findings.md`, `progress.md`
- Create `gemini.md` (Project Constitution)
- Hard stop lifted only after: discovery complete, schema defined, equations locked, blueprint approved

## Phase 1 — B (Blueprint)
- Digital twin scope: reactor heat source, coolant loop, pump system, heat exchanger, sensors (T/P/F)
- State variables: Temperature (T), Pressure (P), Flow (F)
- Equations: use only those defined in `gemini.md`
- Faults: pump failure (eta degradation), overheating (Q_in multiplier), pressure spike (k_p multiplier)
- Determinism: `SEED=42`, fixed `DT=1`, clamp then noise

## Phase 2 — L (Link)
- Validate Python environment
- Validate deps: numpy, pandas, scikit-learn, streamlit

## Phase 3 — A (Architect)
- Docs: `system_design.md`, `simulation_logic.md`, `ml_pipeline.md`
- Code: `simulate_system.py`, `inject_faults.py`, `train_model.py`, `predict.py`, `dashboard.py`

## Phase 4 — Stylize
- Streamlit dashboard: T/P/F graphs + status indicator (NORMAL/WARNING/CRITICAL)
- Execution constraints: run on button click; store results in memory; no infinite loops

## Phase 5 — Trigger
- Run: `streamlit run dashboard.py`

## Done Criteria
- Deterministic outputs with fixed seed
- Fault scenarios visibly affect signals per model
- Isolation Forest runs in real time and flags anomalies
- Entire workflow executes in < 5s for default settings
