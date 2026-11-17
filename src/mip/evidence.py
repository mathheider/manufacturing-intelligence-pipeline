"""Soft-evidence decision layer (v3). Additive: the frozen hard-elimination path
(engine.py + rules.json, preregistered bench-v2) is untouched.

Why: on real photos the hard engine fails by construction. One misread feature
fires a forbidden_if clause and the correct process is gone for good, and the
VLM contradicts itself on exactly the features the rules lean on (``hollow``
samples disagree on ~47% of photos). Here a rule never deletes a process: it
moves its score, scaled by how strongly the perception samples support the
rule's condition. Many weak, partly wrong reads then average out instead of
compounding, so adding rules adds evidence rather than adding ways to fail.

Score(process) = prior_weight * vlm_prior(process)
               + sum over rules r of process:  sign(r) * weight(r) * obs_w(r) * strength(r)

* ``strength`` is the share of usable perception samples that satisfy the
  rule's condition, re-centred so a split vote contributes nothing:
  max(0, (support - tau) / (1 - tau)). Multi-condition rules use the weakest
  condition (min), so a combination is never stronger than its weakest read.
* ``obs_w`` down-weights rules whose feature a photo cannot really show
  (visible 1.0, estimable, not_visible: configurable).
* ``sign`` is -1 for forbid/disfavor and +1 for favor. ``forbid`` is simply
  a strong disfavor here; it only acts as a veto when a caller asks for one.

Every contribution is returned with the rule id and citation, so the per-feature
provenance of the hard engine is kept: each ranking can be explained line by line.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

_OPS = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "gt": lambda a, b: a > b,
    "lt": lambda a, b: a < b,
    "gte": lambda a, b: a >= b,
    "lte": lambda a, b: a <= b,
    "in": lambda a, b: a in b,
}
SIGN = {"forbid": -1.0, "disfavor": -1.0, "favor": +1.0}


@dataclass(frozen=True)
class Cond:
    feature: str
    op: str
    value: Any


@dataclass(frozen=True)
class SoftRule:
    id: str
    process: str
    kind: str                      # forbid | disfavor | favor
    when: tuple                    # tuple[Cond, ...], all must hold
    weight: float
    cite: str = ""
    reason: str = ""
    status: str = ""               # e.g. v2-verified, v2-VERIFY, v3-draft-VERIFY
    observability: str = "visible"  # visible | estimable | not_visible | signature


@dataclass
class Config:
    prior_weight: float = 2.0      # weight on the VLM ranked-list prior
    tau: float = 0.5               # support needed before a rule contributes
    obs_w: Dict[str, float] = field(default_factory=lambda: {
        "visible": 1.0, "signature": 1.0, "estimable": 0.5, "not_visible": 0.25})
    weight_scale: float = 1.0      # global multiplier on rule weights
    min_samples: int = 1           # usable reads needed before a rule may fire
    tie_break: str = "vlm"         # "vlm": VLM rank breaks ties; "alpha": alphabetical (rules-only runs)


def load_rules(path: Path | str) -> tuple:
    """Load a v3 rule file -> (processes, features, rules)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    rules = []
    for r in raw["rules"]:
        conds = tuple(Cond(c["feature"], c["op"], tuple(c["value"]) if isinstance(c["value"], list) else c["value"])
                      for c in r["when"])
        rules.append(SoftRule(r["id"], r["process"], r["kind"], conds, float(r["weight"]),
                              r.get("cite", ""), r.get("reason", ""), r.get("status", ""),
                              r.get("observability", "visible")))
    return raw["processes"], raw["features"], rules


def from_hard_rules(rules_json: Path | str, weight: float = 3.0, observability: Optional[Mapping[str, str]] = None) -> List[SoftRule]:
    """Convert the frozen v2 forbidden_if table to soft rules (same clauses, no veto)."""
    raw = json.loads(Path(rules_json).read_text(encoding="utf-8"))
    out, i = [], 0
    for proc, body in raw["rules"].items():
        for c in body.get("forbidden_if", []):
            i += 1
            obs = (observability or {}).get(c["feature"], "visible")
            out.append(SoftRule(f"V2-{i:03d}", proc, "forbid", (Cond(c["feature"], c["op"], c["value"]),),
                                weight, c.get("cite", ""), c.get("reason", ""),
                                "v2-VERIFY" if "VERIFY" in c.get("cite", "") else "v2-verified", obs))
    return out


def support(samples: Sequence[Any], cond: Cond) -> tuple:
    """Share of usable samples that satisfy ``cond``; (support, n_usable)."""
    fn = _OPS[cond.op]
    usable, hit = 0, 0
    for s in samples:
        if s is None:
            continue
        if cond.op in ("gt", "lt", "gte", "lte") and isinstance(s, bool):
            continue
        usable += 1
        try:
            hit += bool(fn(s, cond.value))
        except TypeError:
            usable -= 1
    return (hit / usable if usable else 0.0), usable


def joint_support(samples: Sequence[Any], conds: Sequence[Cond]) -> tuple:
    """Share of usable samples satisfying ALL ``conds`` (all on one feature).
    A range like 0.8 <= t <= 4 is judged per sample, not per bound."""
    usable, hit = 0, 0
    for s in samples:
        if s is None:
            continue
        ok, valid = True, True
        for c in conds:
            if c.op in ("gt", "lt", "gte", "lte") and isinstance(s, bool):
                valid = False
                break
            try:
                ok = ok and bool(_OPS[c.op](s, c.value))
            except TypeError:
                valid = False
                break
        if not valid:
            continue
        usable += 1
        hit += ok
    return (hit / usable if usable else 0.0), usable


def strength(samples_by_feature: Mapping[str, Sequence[Any]], rule: SoftRule, cfg: Config) -> float:
    """Re-centred support of the rule's condition. Conditions on the same feature
    are evaluated jointly per sample; across features the weakest one counts."""
    by_feat: Dict[str, list] = {}
    for c in rule.when:
        by_feat.setdefault(c.feature, []).append(c)
    vals = []
    for f, conds in by_feat.items():
        s, n = joint_support(samples_by_feature.get(f, ()), conds)
        if n < cfg.min_samples:
            return 0.0
        vals.append(s)
    s = min(vals) if vals else 0.0
    return max(0.0, (s - cfg.tau) / (1.0 - cfg.tau)) if cfg.tau < 1 else float(s >= 1)


def vlm_prior(top8: Sequence[str], processes: Iterable[str]) -> Dict[str, float]:
    """Rank-1 -> 1.0 down to rank-8 -> 0.125; unranked -> 0."""
    pr = {p: 0.0 for p in processes}
    for i, p in enumerate(top8[:8]):
        if p in pr:
            pr[p] = (8 - i) / 8.0
    return pr


def score(processes: Sequence[str], rules: Sequence[SoftRule],
          samples_by_feature: Mapping[str, Sequence[Any]], top8: Sequence[str],
          cfg: Config, explain: bool = False):
    """Return processes ranked best-first (and per-process contributions if explain)."""
    prior = vlm_prior(top8, processes)
    sc = {p: cfg.prior_weight * prior[p] for p in processes}
    why = {p: [] for p in processes} if explain else None
    for r in rules:
        if r.process not in sc:
            continue
        st = strength(samples_by_feature, r, cfg)
        if st <= 0:
            continue
        d = SIGN[r.kind] * r.weight * cfg.weight_scale * cfg.obs_w.get(r.observability, 1.0) * st
        sc[r.process] += d
        if explain:
            why[r.process].append({"rule": r.id, "delta": round(d, 3), "cite": r.cite})
    rank_of = {p: i for i, p in enumerate(top8)} if cfg.tie_break == "vlm" else {}
    ranked = sorted(processes, key=lambda p: (-round(sc[p], 9), rank_of.get(p, 99), p))
    if explain:
        return ranked, sc, why
    return ranked
