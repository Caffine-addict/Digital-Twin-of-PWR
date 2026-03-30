"""Threshold policy constraints and candidate generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ThresholdPair:
    warning: float
    critical: float


def is_valid(pair: ThresholdPair) -> bool:
    if not (-0.5 <= pair.warning <= 0.0):
        return False
    if not (-0.5 <= pair.critical <= 0.0):
        return False
    if not (pair.critical < pair.warning):
        return False
    return True


def candidate_pairs() -> List[ThresholdPair]:
    warnings = [-0.05, -0.1, -0.15, -0.2, -0.25, -0.3]
    criticals = [-0.1, -0.15, -0.2, -0.25, -0.3, -0.35, -0.4]
    out: List[ThresholdPair] = []
    for w in warnings:
        for c in criticals:
            p = ThresholdPair(warning=float(w), critical=float(c))
            if is_valid(p):
                out.append(p)
    return out
