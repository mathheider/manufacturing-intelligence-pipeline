"""Consistency-calibrated commitment.

Query a feature ``n`` times, then decide whether the answer is stable enough to
commit. Numeric features must have relative standard deviation within
``tol_rel``; categorical/boolean features must have a majority fraction of at
least ``min_agree``. Unstable reads are *skipped*, not guessed -- this is what
prevents noisy single reads from causing false eliminations.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Any, List


@dataclass
class ConsistencyResult:
    value: Any
    uncertainty: float          # numeric: stdev; categorical: 1 - majority frac
    n_samples: int              # number of *valid* (non-None) samples
    stable: bool
    raw_samples: List[Any]


def sample_numeric(samples: List[float], tol_rel: float = 0.10,
                   abs_tol: float = 1e-6) -> ConsistencyResult:
    if not samples:
        return ConsistencyResult(None, float("inf"), 0, False, [])
    mean = statistics.fmean(samples)
    stdev = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    # Perfect (or near-perfect) agreement is stable regardless of the mean.
    # This matters for genuinely-zero features (e.g. draft angle 0 deg): the
    # relative-stdev test alone would divide by a zero mean and mark a unanimous
    # reading as maximally unstable, so the agent could never commit it.
    if stdev <= abs_tol:
        stable = True
    elif mean:
        stable = (stdev / abs(mean)) <= tol_rel
    else:
        stable = False
    return ConsistencyResult(round(mean, 3), round(stdev, 3), len(samples),
                             stable, list(samples))


def sample_categorical(samples: List[Any], min_agree: float = 0.60) -> ConsistencyResult:
    if not samples:
        return ConsistencyResult(None, 1.0, 0, False, [])
    top_val, top_n = Counter(samples).most_common(1)[0]
    frac = top_n / len(samples)
    return ConsistencyResult(top_val, round(1.0 - frac, 3), len(samples),
                             frac >= min_agree, list(samples))


def run_consistency(vlm, image, question: str, feature_name: str,
                    feature_type: str, n: int = 5, tol_rel: float = 0.10,
                    min_agree: float = 0.60) -> ConsistencyResult:
    samples = []
    for _ in range(n):
        resp = vlm.ask(image, question, feature_name, feature_type)
        if resp.parsed is not None:
            samples.append(resp.parsed)
    if feature_type == "numeric":
        return sample_numeric(samples, tol_rel=tol_rel)
    return sample_categorical(samples, min_agree=min_agree)
