# -*- coding: utf-8 -*-
"""ADDITIVE per-instance record dump of the COMPLETED preregistered bench-v2 run.

Replays the full 635-photo closed pool against the archived response cache
(cache-only backend: any real call raises, so no API call can occur and no
number can drift). Writes one record per instance x method to
outputs/records_real_v2.jsonl.

VERIFICATION GATE: after dumping, this script rebuilds the aggregate summary
with the exact seed parts / n_boot of the frozen run and asserts it EQUALS
the preregistered artifact outputs/benchmark_real_v2.json (all_components +
hard_subset, every field). Records are only trustworthy if the gate passes.

This artifact is ADDITIVE (metrics rework, 2026-09-01, METRICS_MEMO.md):
benchmark_real_v2.json is never rewritten; H1-H3 remain judged on it alone.

    python scripts/dump_records_v2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

MIP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MIP / "src"))

from mip.benchmark import _run_all_pipelines, _score_result, _observed_features
from mip.config import Config
from mip.dataset import load_real
from mip.engine import shortlist
from mip.metrics import MethodAggregate, summarize
from mip.pipelines import ALL_METHODS, TOPK_METHOD
from mip.rules import RuleSet
from mip.vlm.cache import CachedVLM, PROMPT_VERSION


class CacheMiss(Exception):
    pass


class _CacheOnly:
    _model = "cache-only record dump (no network)"

    def ask(self, *a, **k):
        raise CacheMiss()

    def ask_process(self, *a, **k):
        raise CacheMiss()

    def ask_process_topk(self, *a, **k):
        raise CacheMiss()


def _ranked(method, res, rules, inst, k=8):
    if res.extra.get("ranked"):
        return list(res.extra["ranked"])[:k]
    if res.graph is not None or method == "RuleSet":
        return shortlist(res.predicted, rules,
                         _observed_features(method, res, inst), k)
    return sorted(res.predicted)[:k]


def main() -> int:
    assert PROMPT_VERSION == "bench-v2", "prompt version drifted"
    cfg = Config()
    rules = RuleSet.load(None)
    cache_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        MIP / "outputs" / "vlm_cache" / "bench-gemini-2.5-flash-lite")
    print(f"cache dir: {cache_dir}")
    instances = load_real(stratified=True)
    methods = [m for m in ALL_METHODS if m != "RuleSet"]
    K, SK = 8, 3
    rows = methods + [TOPK_METHOD]
    agg = {m: MethodAggregate() for m in rows}
    agg_hard = {m: MethodAggregate() for m in rows}
    records, skipped = [], 0

    for inst in instances:
        if not inst.image or not (MIP / inst.image).exists():
            continue
        vlm = CachedVLM(_CacheOnly(), cache_dir)
        try:
            res = _run_all_pipelines(inst, rules, cfg, "gemini", vlm, methods, K)
        except CacheMiss:
            skipped += 1
            continue
        hard = inst.expert_processes != {"Injection_Molding"}
        for m, r in res.items():
            s, contained, size, sf1 = _score_result(m, r, inst, rules, SK)
            agg[m].add(s, r.n_probes, contained, size, sf1)
            if hard:
                agg_hard[m].add(s, r.n_probes, contained, size, sf1)
            records.append({
                "id": inst.id,
                "category": inst.archetype,
                "hard": hard,
                "method": m,
                "expert": sorted(inst.expert_processes),
                "predicted": sorted(r.predicted),
                "ranked": _ranked(m, r, rules, inst, K),
                "n_probes": r.n_probes,
                "precision": round(s.precision, 4),
                "recall": round(s.recall, 4),
                "f1": round(s.f1, 4),
                "fp": s.fp, "fn": s.fn, "tp": s.tp,
                "contained": contained,
                "set_size": size,
                "shortlist_f1": round(sf1, 4),
            })

    n_proc = len(rules.processes)
    rebuilt_all = {m: summarize(agg[m], ("all", m), 1000, n_proc) for m in rows}
    rebuilt_hard = {m: summarize(agg_hard[m], ("hard", m), 1000, n_proc) for m in rows}

    frozen = json.loads((MIP / "outputs" / "benchmark_real_v2.json").read_text(encoding="utf-8"))
    gate_all = frozen["all_components"] == rebuilt_all
    gate_hard = frozen["hard_subset"] == rebuilt_hard
    print(f"instances scored: {len(records)//len(rows)}  skipped: {skipped}")
    print(f"VERIFICATION GATE all_components: {'PASS' if gate_all else 'FAIL'}")
    print(f"VERIFICATION GATE hard_subset:    {'PASS' if gate_hard else 'FAIL'}")
    if not (gate_all and gate_hard):
        for m in rows:
            for blk, mine, theirs in (("all", rebuilt_all, frozen["all_components"]),
                                      ("hard", rebuilt_hard, frozen["hard_subset"])):
                if mine[m] != theirs[m]:
                    diffs = {k: (theirs[m].get(k), mine[m].get(k))
                             for k in set(mine[m]) | set(theirs[m])
                             if mine[m].get(k) != theirs[m].get(k)}
                    print(f"  MISMATCH {blk}/{m}: {diffs}")
        print("records NOT written (gate failed)")
        return 1

    out = MIP / "outputs" / "records_real_v2.jsonl"
    with out.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"_meta": {
            "source": "cache-only replay of completed bench-v2 (no API calls)",
            "verified_against": "outputs/benchmark_real_v2.json (exact match, all fields)",
            "prompt_version": PROMPT_VERSION, "direct_topk": K, "shortlist_k": SK,
            "n_clauses": rules.n_clauses, "date": "2026-09-01"}}) + "\n")
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"wrote {out} ({len(records)} records)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
