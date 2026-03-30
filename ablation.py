"""Ablation comparison of system variants."""

from __future__ import annotations

from typing import Any, Dict, Optional

from experiments import run_all


def compare_variants(
    *,
    base: Dict[str, Any],
    variant: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare two experiment result sets."""
    # Expect base/variant as {"results": [...]}
    b = base.get("results", [])
    v = variant.get("results", [])
    by_name = {r["scenario"]["name"]: r for r in b}

    diffs = []
    for r in v:
        name = r["scenario"]["name"]
        rb = by_name.get(name)
        if not rb:
            continue
        mb = rb["metrics"]
        mv = r["metrics"]
        diffs.append(
            {
                "scenario": name,
                "delta_false_positive_rate": float(mv.get("false_positive_rate", 0.0) - mb.get("false_positive_rate", 0.0)),
                "delta_detection_rate": float(mv.get("detection_rate", 0.0) - mb.get("detection_rate", 0.0)),
                "delta_detection_delay": (
                    None
                    if (mv.get("detection_delay") is None or mb.get("detection_delay") is None)
                    else float(mv.get("detection_delay") - mb.get("detection_delay"))
                ),
            }
        )
    return {"diffs": diffs}


def run_ablation(
    *,
    model_path: str,
    base_thresholds: Optional[Dict[str, float]] = None,
    variant_thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    base = run_all(model_path=model_path, thresholds=base_thresholds)
    variant = run_all(model_path=model_path, thresholds=variant_thresholds)
    return {
        "base": base,
        "variant": variant,
        "comparison": compare_variants(base=base, variant=variant),
    }
