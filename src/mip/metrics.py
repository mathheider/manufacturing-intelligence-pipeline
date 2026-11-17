"""Scoring and uncertainty.

Set-valued prediction vs. set-valued expert ground truth, scored per instance
then macro-averaged. Adds a deterministic bootstrap 95% confidence interval on
macro-F1 -- the "printed confidence interval" the prelim handoff lists as an
open item. The bootstrap is seeded, so the interval is reproducible.

Alongside set-F1 the aggregate carries three elimination-native metrics
(CLAUDE_CODE_GOALS.md goal 4), because an elimination engine's job is *safe
narrowing*, which set-F1 alone understates:

* expert-containment rate -- fraction of instances whose full expert set
  survived (no false elimination at all);
* mean survivor-set size -- how far the 17-process space was narrowed
  (reduction = 17 - size);
* shortlist F1@k -- F1 of the top-k survivors under the symbolic
  constraint-evidence ranking in ``engine.shortlist`` (rule-scored, cited;
  the categorical call never goes back to the net).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set, Tuple

from ._seed import rng


@dataclass
class Score:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int          # false eliminations of an expert-feasible process


def score(predicted: Set[str], truth: Set[str]) -> Score:
    tp = len(predicted & truth)
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return Score(p, r, f, tp, fp, fn)


@dataclass
class MethodAggregate:
    f1: List[float] = field(default_factory=list)
    precision: List[float] = field(default_factory=list)
    recall: List[float] = field(default_factory=list)
    probes: List[float] = field(default_factory=list)
    false_elims: List[float] = field(default_factory=list)
    containment: List[float] = field(default_factory=list)
    set_size: List[float] = field(default_factory=list)
    shortlist_f1: List[float] = field(default_factory=list)

    def add(self, s: Score, probes: int, contained: bool = False,
            set_size: int = 0, shortlist_f1: float = 0.0) -> None:
        self.f1.append(s.f1)
        self.precision.append(s.precision)
        self.recall.append(s.recall)
        self.probes.append(probes)
        self.false_elims.append(s.fn)
        self.containment.append(1.0 if contained else 0.0)
        self.set_size.append(set_size)
        self.shortlist_f1.append(shortlist_f1)

    def __len__(self) -> int:
        return len(self.f1)


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def bootstrap_ci(values: Sequence[float], seed_parts, n_boot: int = 1000,
                 alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``values``. Deterministic."""
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    r = rng("bootstrap", *seed_parts)
    means = []
    for _ in range(n_boot):
        resample = [values[r.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (round(lo, 3), round(hi, 3))


def summarize(agg: MethodAggregate, seed_parts, n_boot: int = 1000,
              n_processes: int = 17) -> Dict:
    lo, hi = bootstrap_ci(agg.f1, seed_parts, n_boot=n_boot)
    avg_size = _mean(agg.set_size)
    return {
        "precision": round(_mean(agg.precision), 3),
        "recall": round(_mean(agg.recall), 3),
        "f1": round(_mean(agg.f1), 3),
        "f1_ci95": [lo, hi],
        "avg_probes": round(_mean(agg.probes), 3),
        "avg_false_elims": round(_mean(agg.false_elims), 3),
        "containment": round(_mean(agg.containment), 3),
        "avg_set_size": round(avg_size, 3),
        "survivor_reduction": round(n_processes - avg_size, 3),
        "shortlist_f1": round(_mean(agg.shortlist_f1), 3),
        "n_instances": len(agg),
    }
