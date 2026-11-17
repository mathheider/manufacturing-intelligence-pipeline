# -*- coding: utf-8 -*-
"""Disclosed INTERIM readout of the IN-PROGRESS preregistered bench-v2 run.

What this is: the run cache already holds complete response sets for a prefix
of the closed pool. Because the run was launched with --stratified, ANY prefix
of the instance list is category-balanced by construction (see
dataset.load_real). This script scores ONLY photos whose every response is
already archived, using the UNMODIFIED mip pipeline code: the VLM is
cache-only, and any cache miss raises, skipping that photo. No API call can
occur, no phrasing/metric/threshold differs from the frozen design.

What this is NOT: the preregistered result. H1-H3 are judged SOLELY on the
completed closed pool (n=635, benchmark_real_v2.json). This file is labeled
INTERIM everywhere it is cited and ships alongside the final number whichever
of the two is better. Deviation logged in PREREGISTRATION_BENCH_V2.md
execution log, 2026-08-05, before this script was first run.

    python scripts/score_v2_interim.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

MIP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MIP / "src"))

from mip.benchmark import _run_all_pipelines, _score_result   # frozen path
from mip.config import Config
from mip.dataset import load_real
from mip.metrics import MethodAggregate, summarize
from mip.pipelines import ALL_METHODS, TOPK_METHOD
from mip.rules import RuleSet
from mip.vlm.cache import CachedVLM, PROMPT_VERSION


class CacheMiss(Exception):
    pass


class _CacheOnly:
    """Inner backend that refuses every real call: replay-only."""
    _model = "cache-only interim replay (no network)"

    def ask(self, *a, **k):
        raise CacheMiss()

    def ask_process(self, *a, **k):
        raise CacheMiss()

    def ask_process_topk(self, *a, **k):
        raise CacheMiss()


def main() -> int:
    assert PROMPT_VERSION == "bench-v2", "prompt version drifted"
    cfg = Config()                      # same defaults as the running command
    rules = RuleSet.load(None)
    instances = load_real(stratified=True)
    pool_size = len(instances)
    methods = [m for m in ALL_METHODS if m != "RuleSet"]  # no oracle on photos
    K, SK = 8, 3
    rows = methods + [TOPK_METHOD]
    agg = {m: MethodAggregate() for m in rows}
    agg_hard = {m: MethodAggregate() for m in rows}
    scored, skipped, cats = [], 0, {}

    for inst in instances:
        if not inst.image or not (MIP / inst.image).exists():
            continue
        # per-model cache dir: the corrected launch passes --model
        # gemini-2.5-flash-lite, and benchmark.py names the dir bench-<model>
        vlm = CachedVLM(_CacheOnly(),
                        MIP / "outputs" / "vlm_cache" / "bench-gemini-2.5-flash-lite")
        try:
            res = _run_all_pipelines(inst, rules, cfg, "gemini", vlm,
                                     methods, K)
        except CacheMiss:
            skipped += 1
            continue
        hard = inst.expert_processes != {"Injection_Molding"}
        for m, r in res.items():
            s, contained, size, sf1 = _score_result(m, r, inst, rules, SK)
            agg[m].add(s, r.n_probes, contained, size, sf1)
            if hard:
                agg_hard[m].add(s, r.n_probes, contained, size, sf1)
        scored.append(inst.id)
        cats[inst.archetype] = cats.get(inst.archetype, 0) + 1

    n_proc = len(rules.processes)
    out = {
        "INTERIM": True,
        "note": ("Disclosed interim readout of the IN-PROGRESS preregistered "
                 "bench-v2 run: cache-complete photos only, scored once with "
                 "the frozen pipeline code (cache-only VLM; any miss skips "
                 "the photo). Stratified launch order makes this prefix "
                 "category-balanced by construction. H1-H3 are judged SOLELY "
                 "on the completed closed pool (n=635, "
                 "benchmark_real_v2.json); this file never replaces it and "
                 "ships alongside it regardless of which is better. "
                 "Deviation logged in PREREGISTRATION_BENCH_V2.md, "
                 "2026-08-05."),
        "backend": "gemini",
        "dataset": "real",
        "dataset_size": len(scored),
        "pool_size": pool_size,
        "skipped_incomplete_or_pending": skipped,
        "categories_covered": len(cats),
        "per_category_counts": dict(sorted(cats.items())),
        "trials": 1,
        "direct_topk": K,
        "shortlist_k": SK,
        "n_clauses": rules.n_clauses,
        "prompt_version": PROMPT_VERSION,
        "config": cfg.to_dict(),
        "all_components": {m: summarize(agg[m], ("all", m), 1000, n_proc)
                           for m in rows},
        "hard_subset": {m: summarize(agg_hard[m], ("hard", m), 1000, n_proc)
                        for m in rows},
        "scored_instances": scored,
    }
    p = MIP / "outputs" / "benchmark_real_v2_interim.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {p}")
    print(f"n_scored={len(scored)} of pool {pool_size} | "
          f"categories covered: {len(cats)} | skipped/pending: {skipped}")
    for sub in ("all_components", "hard_subset"):
        for m in ("Direct", TOPK_METHOD, "Hybrid"):
            r = out[sub].get(m)
            if r:
                print(f"  {sub:15s} {m:10s} F1 {r['f1']:.3f} "
                      f"CI [{r['f1_ci95'][0]:.3f},{r['f1_ci95'][1]:.3f}] "
                      f"containment {r['containment']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
