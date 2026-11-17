"""Benchmark instances.

Two sources, both fully offline and reproducible:

* ``load_seed()`` -- the 14 hand-labeled polymer-composite automotive
  components (the authoritative set; image paths for a real VLM).

* ``generate(count, seed)`` -- a deterministic synthetic expansion to any size
  (the prelim's "1,000-instance benchmark" scale). Each synthetic instance is a
  geometric variation of one seed archetype: continuous and integer features are
  jittered within plausible bands while the class-defining booleans (hollow,
  axisymmetric, ...) are held fixed. Every variation is checked against the
  oracle rule engine so that *no expert-feasible process is ever eliminated
  under perfect perception* -- i.e. recall is never capped by the data. This is
  synthetic, controlled data, NOT a real expert survey; it exists to measure how
  perception and decision logic behave at scale, not to add new ground truth.

Class imbalance is intentional: the seed set is injection-molding-heavy, and the
round-robin expansion preserves that skew -- the corpus-frequency bias the
architecture is designed to neutralise.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

from .engine import eliminate
from ._seed import rng
from .rules import RuleSet

_COMPONENTS = Path(__file__).resolve().parents[2] / "data" / "components.json"


@dataclass
class Instance:
    id: str
    archetype: str
    features: Dict[str, object]
    expert_processes: Set[str]
    image: Optional[str] = None
    synthetic: bool = False


def load_seed(path: Path | str | None = None) -> List[Instance]:
    rows = json.loads(Path(path or _COMPONENTS).read_text(encoding="utf-8"))
    return [Instance(id=r["id"], archetype=r["id"], features=dict(r["features"]),
                     expert_processes=set(r["expert_processes"]),
                     image=r.get("image")) for r in rows]


def _feasible(features, expert: Set[str], rules: RuleSet) -> bool:
    """True iff every expert process survives the oracle engine on ``features``."""
    survivors, _ = eliminate(rules, features)
    return expert.issubset(set(survivors))


def _perturb(base: Dict[str, object], rules: RuleSet, r, scale: float) -> Dict[str, object]:
    out = dict(base)
    for name, fdef in rules.features.items():
        if name not in base:
            continue
        if fdef.type == "numeric":
            lo, hi = (fdef.range or (0.0, float(base[name]) * 2 + 1.0))
            val = float(base[name]) * (1.0 + r.uniform(-scale, scale))
            out[name] = round(min(hi, max(lo, val)), 3)
        elif fdef.type == "integer":
            lo, hi = (fdef.range or (0, int(base[name]) + 2))
            delta = r.randint(-1, 1) if scale > 0 else 0
            out[name] = int(min(hi, max(lo, int(base[name]) + delta)))
        # booleans are class-defining: held fixed
    return out


def generate(count: int, seed: int = 0, path: Path | str | None = None) -> List[Instance]:
    rules = RuleSet.load()
    archetypes = load_seed(path)
    instances: List[Instance] = []
    for i in range(count):
        arch = archetypes[i % len(archetypes)]
        r = rng(seed, "gen", i)
        # Shrink the jitter window until the feasibility invariant holds.
        feats = dict(arch.features)
        for scale in (0.20, 0.12, 0.06, 0.0):
            cand = _perturb(arch.features, rules, r, scale)
            if _feasible(cand, arch.expert_processes, rules):
                feats = cand
                break
        instances.append(Instance(
            id=f"{arch.id}__{i:04d}", archetype=arch.id, features=feats,
            expert_processes=set(arch.expert_processes),
            image=arch.image, synthetic=True))
    return instances


def load_dataset(size: int = 0, seed: int = 0) -> List[Instance]:
    """size<=0 or size==14 -> the seed set; otherwise a synthetic set of `size`."""
    seed_set = load_seed()
    if size <= 0 or size == len(seed_set):
        return seed_set
    if size < len(seed_set):
        return seed_set[:size]
    return generate(size, seed=seed)


_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST = _ROOT / "images_pool" / "images_manifest.json"
_QC_REPORT = _ROOT / "outputs" / "qc_report.json"


def load_real(manifest: Path | str | None = None,
              qc_report: Path | str | None = None,
              accepted_only: bool = True,
              stratified: bool = False) -> List[Instance]:
    """Instances for the real CC-licensed photo pool (goal 8).

    HONESTY NOTE (mirrored in docs/IMPROVEMENTS.md): expert labels here are
    CATEGORY-LEVEL sets -- the processes feasible for the part category (from
    scripts/image_categories.py, derived from the literature, independent of
    any model) -- not per-instance expert annotations. ``features`` is empty:
    real photos have no ground-truth feature oracle, so the RuleSet
    (perfect-perception) pipeline is undefined on this dataset and the
    benchmark skips it.

    If a QC report exists (scripts/qc_images.py), only VLM-QC-accepted images
    are included; images missing from disk are skipped either way.
    """
    mpath = Path(manifest) if manifest else _MANIFEST
    entries = json.loads(mpath.read_text(encoding="utf-8"))["images"]
    accepted = None
    qpath = Path(qc_report) if qc_report else _QC_REPORT
    if accepted_only and qpath.exists():
        rep = json.loads(qpath.read_text(encoding="utf-8"))
        accepted = {r["path"] for r in rep.get("results", []) if r.get("accept")}
    out: List[Instance] = []
    for e in entries:
        if not (_ROOT / e["path"]).exists():
            continue
        if accepted is not None and e["path"] not in accepted:
            continue
        out.append(Instance(
            id=Path(e["path"]).stem, archetype=e["category"], features={},
            expert_processes=set(e["expert_processes"]),
            image=e["path"], synthetic=False))
    if stratified:
        # Round-robin interleave categories (deterministic: manifest order
        # within category, alphabetical across categories) so that ANY prefix
        # of the list is category-balanced -- makes --limit N a stratified
        # subsample instead of a run of whichever category was fetched first.
        from collections import OrderedDict, deque
        by_cat: "OrderedDict[str, deque]" = OrderedDict()
        for inst in out:
            by_cat.setdefault(inst.archetype, deque()).append(inst)
        queues = [by_cat[k] for k in sorted(by_cat)]
        mixed: List[Instance] = []
        while queues:
            queues = [q for q in queues if q]
            for q in queues:
                if q:
                    mixed.append(q.popleft())
        return mixed
    return out
