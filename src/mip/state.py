"""Constraint-graph state: the structured symbolic output of the pipeline.

A ``ConstraintGraph`` is the machine-checkable artifact the thesis is built on.
It records, for one component:

* ``features``    -- every feature the agent observed, with its committed value,
                     calibrated uncertainty, sample count, and commit/skip flag.
* ``survivors``   -- the process candidates that no rule eliminated.
* ``eliminations``-- for each eliminated process, the *cited* provenance of the
                     elimination (which feature observation fired which rule,
                     traced to its literature source).
* ``trace``       -- an ordered log of every planning / probing / pruning event.

Everything is JSON-serialisable via :meth:`to_dict`, so an external checker can
audit a decision without importing this package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class FeatureObservation:
    """One perceived feature value plus its calibrated uncertainty."""

    name: str
    value: Any
    uncertainty: float          # numeric: stdev; categorical: 1 - majority frac
    n_samples: int
    committed: bool             # False => perception was too unstable to use
    raw_samples: List[Any] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "value": self.value,
            "uncertainty": self.uncertainty,
            "n_samples": self.n_samples,
            "committed": self.committed,
            "raw_samples": list(self.raw_samples),
        }


@dataclass
class Elimination:
    """The cited provenance of a single process elimination."""

    process: str
    feature: str
    op: str
    value: Any
    observed: Any
    cite: str
    reason: str

    def as_text(self) -> str:
        return (f"{self.feature} {self.op} {self.value} "
                f"(observed {self.observed}) [{self.cite} - {self.reason}]")

    def to_dict(self) -> Dict:
        return {
            "process": self.process, "feature": self.feature, "op": self.op,
            "threshold": self.value, "observed": self.observed,
            "cite": self.cite, "reason": self.reason, "text": self.as_text(),
        }


@dataclass
class ConstraintGraph:
    features: Dict[str, FeatureObservation] = field(default_factory=dict)
    survivors: List[str] = field(default_factory=list)
    eliminations: Dict[str, Elimination] = field(default_factory=dict)
    trace: List[Dict] = field(default_factory=list)

    def commit(self, obs: FeatureObservation) -> None:
        self.features[obs.name] = obs

    def log(self, event: Dict) -> None:
        self.trace.append(event)

    def committed_features(self) -> Dict[str, FeatureObservation]:
        return {k: v for k, v in self.features.items() if v.committed}

    def n_committed(self) -> int:
        return sum(1 for v in self.features.values() if v.committed)

    def to_dict(self) -> Dict:
        return {
            "features": {k: v.to_dict() for k, v in self.features.items()},
            "survivors": list(self.survivors),
            "eliminations": {k: v.to_dict() for k, v in self.eliminations.items()},
            "trace": list(self.trace),
        }
