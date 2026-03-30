from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class HybridCorrector:
    # affine correction: corrected = x + clip(W*x + b, -5, 5)
    W: np.ndarray  # shape (3,3)
    b: np.ndarray  # shape (3,)

    def correct(self, T: float, P: float, F: float) -> Tuple[float, float, float]:
        x = np.asarray([T, P, F], dtype=float)
        delta = self.W @ x + self.b
        # clamp correction delta as required
        delta = np.clip(delta, -5.0, 5.0)
        y = x + delta
        return float(y[0]), float(y[1]), float(y[2])


def fit_hybrid_corrector(T_true, P_true, F_true, T_meas, P_meas, F_meas) -> HybridCorrector:
    """
    Fit an affine model for residuals: (meas-true) ~ W*[T,P,F] + b
    Uses least squares.
    """
    T_true = np.asarray(T_true, dtype=float)
    P_true = np.asarray(P_true, dtype=float)
    F_true = np.asarray(F_true, dtype=float)
    T_meas = np.asarray(T_meas, dtype=float)
    P_meas = np.asarray(P_meas, dtype=float)
    F_meas = np.asarray(F_meas, dtype=float)

    X = np.column_stack([T_true, P_true, F_true, np.ones_like(T_true, dtype=float)])  # (n,4)
    Y = np.column_stack([T_meas - T_true, P_meas - P_true, F_meas - F_true])  # (n,3)

    B, *_ = np.linalg.lstsq(X, Y, rcond=None)  # (4,3)
    W = B[:3, :].T  # (3,3)
    b = B[3, :]  # (3,)
    return HybridCorrector(W=W, b=b)
