"""Simple bandit agent for adaptive threshold selection.

Optional and disabled by default.

Determinism:
- Selection is deterministic (no RNG) so enabling this does not introduce nondeterminism.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
import json

from threshold_policy import ThresholdPair, candidate_pairs


@dataclass
class BanditAgent:
    epsilon: float
    counts: List[int]
    values: List[float]


def new_agent(*, epsilon: float = 0.1) -> BanditAgent:
    n = len(candidate_pairs())
    return BanditAgent(epsilon=float(epsilon), counts=[0] * n, values=[0.0] * n)


def select_action(agent: BanditAgent) -> int:
    n = len(agent.values)
    if n == 0:
        return 0
    # Deterministic epsilon-greedy:
    # - "explore" picks a round-robin arm based on total pulls
    # - otherwise exploit best value
    total_pulls = int(sum(agent.counts))
    explore = (float(agent.epsilon) > 0.0) and ((total_pulls % 10) == 0)
    if explore:
        return int(total_pulls % n)
    return int(max(range(n), key=lambda i: float(agent.values[i])))


def get_thresholds(action: int) -> ThresholdPair:
    pairs = candidate_pairs()
    return pairs[int(action)]


def update(agent: BanditAgent, action: int, reward: float) -> None:
    i = int(action)
    agent.counts[i] += 1
    n = agent.counts[i]
    agent.values[i] = float(agent.values[i] + (float(reward) - float(agent.values[i])) / float(n))


def save(agent: BanditAgent, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"epsilon": agent.epsilon, "counts": agent.counts, "values": agent.values}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load(path: Path) -> BanditAgent:
    p = Path(path)
    if not p.exists():
        return new_agent()
    payload = json.loads(p.read_text(encoding="utf-8"))
    return BanditAgent(
        epsilon=float(payload.get("epsilon", 0.1)),
        counts=list(payload.get("counts", [])),
        values=list(payload.get("values", [])),
    )
