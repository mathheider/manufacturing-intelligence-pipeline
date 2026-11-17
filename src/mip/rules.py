"""Rule table: load, *validate*, and query the feasibility rule set.

The original prototype passed a raw ``yaml.safe_load`` dict around and trusted
it. That is a reproducibility hazard: a typo in a feature name, an unknown
operator, or a boolean threshold on a numeric feature would silently produce
wrong eliminations. Here the rule table is parsed into typed dataclasses and
validated up front, so a malformed rule table fails loudly at load time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

VALID_OPS = frozenset({"eq", "ne", "gt", "lt", "gte", "lte"})
NUMERIC_OPS = frozenset({"gt", "lt", "gte", "lte", "eq", "ne"})
BOOL_OPS = frozenset({"eq", "ne"})

_DEFAULT_RULES = Path(__file__).resolve().parents[2] / "data" / "rules.json"


@dataclass(frozen=True)
class Feature:
    name: str
    type: str                       # numeric | integer | boolean
    unit: str = ""
    range: tuple = ()
    question: str = ""              # explicit VLM question (else templated)


@dataclass(frozen=True)
class Clause:
    """A single forbidden_if predicate with its literature provenance."""

    feature: str
    op: str
    value: Any
    cite: str
    reason: str


class RuleSet:
    def __init__(self, processes: List[str], features: Dict[str, Feature],
                 rules: Dict[str, List[Clause]]):
        self.processes = list(processes)
        self.features = features
        self.rules = rules
        self._validate()

    # -- construction --------------------------------------------------------

    @classmethod
    def load(cls, path: Path | str | None = None) -> "RuleSet":
        path = Path(path) if path else _DEFAULT_RULES
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        features = {
            name: Feature(name=name, type=spec["type"],
                          unit=spec.get("unit", ""),
                          range=tuple(spec.get("range", ())),
                          question=spec.get("question", ""))
            for name, spec in raw["features"].items()
        }
        rules = {
            proc: [Clause(c["feature"], c["op"], c["value"],
                          c.get("cite", ""), c.get("reason", ""))
                   for c in body.get("forbidden_if", [])]
            for proc, body in raw["rules"].items()
        }
        return cls(raw["processes"], features, rules)

    # -- validation ----------------------------------------------------------

    def _validate(self) -> None:
        for proc in self.processes:
            if proc not in self.rules:
                # a process may legitimately have zero rules; normalise to []
                self.rules.setdefault(proc, [])
        unknown = set(self.rules) - set(self.processes)
        if unknown:
            raise ValueError(f"rules reference unknown processes: {sorted(unknown)}")

        for proc, clauses in self.rules.items():
            for c in clauses:
                if c.feature not in self.features:
                    raise ValueError(
                        f"{proc}: rule references unknown feature {c.feature!r}")
                if c.op not in VALID_OPS:
                    raise ValueError(f"{proc}: unknown operator {c.op!r}")
                ftype = self.features[c.feature].type
                if ftype in ("numeric", "integer"):
                    if c.op not in NUMERIC_OPS:
                        raise ValueError(
                            f"{proc}: op {c.op!r} invalid for numeric {c.feature}")
                    if isinstance(c.value, bool) or not isinstance(c.value, (int, float)):
                        raise ValueError(
                            f"{proc}: numeric feature {c.feature} needs a numeric "
                            f"threshold, got {c.value!r}")
                elif ftype == "boolean":
                    if c.op not in BOOL_OPS:
                        raise ValueError(
                            f"{proc}: boolean feature {c.feature} only supports "
                            f"eq/ne, got {c.op!r}")
                    if not isinstance(c.value, bool):
                        raise ValueError(
                            f"{proc}: boolean feature {c.feature} needs a bool "
                            f"threshold, got {c.value!r}")

    # -- queries -------------------------------------------------------------

    def clauses_for(self, process: str) -> List[Clause]:
        return self.rules.get(process, [])

    def features_referenced_by(self, process: str) -> set:
        return {c.feature for c in self.clauses_for(process)}

    @property
    def n_clauses(self) -> int:
        return sum(len(v) for v in self.rules.values())

    def __repr__(self) -> str:
        return (f"RuleSet({len(self.processes)} processes, "
                f"{len(self.features)} features, {self.n_clauses} clauses)")
