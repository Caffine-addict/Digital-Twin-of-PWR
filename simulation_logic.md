# Simulation Logic

## Tick Execution Order (STRICT)
1. Apply faults
2. Compute flow
3. Compute temperature
4. Compute pressure
5. Clamp
6. Add noise -> measured values
7. Store

## Notes
- `DT = 1` is fixed.
- Faults modify effective inputs/coefficients for the current tick (e.g., `Q_in`, `eta_pump`, `k_p`) and then the locked equations are applied.
- Clamping is applied to true values before sensor noise.
