# -*- coding: utf-8 -*-
"""H1 tally for the completed bench-v2 run: false eliminations by feature.

H1 (PREREGISTRATION_BENCH_V2.md): bench-v2 phrasing reduces false
eliminations attributable to the hollow read by >=50% relative to v1
(v1 baseline: 58% of 281 false elims over 150 photos = 163, i.e. 1.087/part).
Pools differ (150 vs 635), so the preregistered comparison is rate-based:
hollow-attributed false elims per part. Cache-only replay of the completed
archive; no API call possible; frozen pipeline code.
"""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path
MIP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MIP / "src"))
from mip.benchmark import _run_all_pipelines
from mip.config import Config
from mip.dataset import load_real
from mip.rules import RuleSet
from mip.vlm.cache import CachedVLM, PROMPT_VERSION

class CacheMiss(Exception): pass
class _CacheOnly:
    _model = "cache-only replay"
    def ask(self, *a, **k): raise CacheMiss()
    def ask_process(self, *a, **k): raise CacheMiss()
    def ask_process_topk(self, *a, **k): raise CacheMiss()

assert PROMPT_VERSION == "bench-v2"
cfg = Config(); rules = RuleSet.load(None)
instances = load_real(stratified=True)
CACHE = MIP / "outputs" / "vlm_cache" / "bench-gemini-2.5-flash-lite"
feat_all, feat_hard = Counter(), Counter()
n_scored = n_skip = 0; total_fe_all = total_fe_hard = 0; n_hard = 0
for inst in instances:
    if not inst.image or not (MIP / inst.image).exists(): continue
    vlm = CachedVLM(_CacheOnly(), CACHE)
    try:
        res = _run_all_pipelines(inst, rules, cfg, "gemini", vlm,
                                 ["Direct", "Greedy", "Hybrid"], 8)
    except CacheMiss:
        n_skip += 1; continue
    n_scored += 1
    hard = inst.expert_processes != {"Injection_Molding"}
    if hard: n_hard += 1
    g = res["Hybrid"].graph
    for proc, elim in g.eliminations.items():
        if proc in inst.expert_processes:          # false elimination
            feat_all[elim.feature] += 1; total_fe_all += 1
            if hard:
                feat_hard[elim.feature] += 1; total_fe_hard += 1
print(f"scored {n_scored} (hard {n_hard}), skipped {n_skip}")
print(f"Hybrid false elims ALL: {total_fe_all}  ({total_fe_all/max(1,n_scored):.3f}/part)")
for f, c in feat_all.most_common():
    print(f"  {f:28s} {c:5d}  {100*c/max(1,total_fe_all):5.1f}%  {c/max(1,n_scored):.3f}/part")
print(f"Hybrid false elims HARD: {total_fe_hard}  ({total_fe_hard/max(1,n_hard):.3f}/part)")
for f, c in feat_hard.most_common():
    print(f"  {f:28s} {c:5d}  {100*c/max(1,total_fe_hard):5.1f}%")
V1_HOLLOW_PER_PART = 163 / 150.0
hollow_feats = [f for f in feat_all if "hollow" in f.lower()]
h_count = sum(feat_all[f] for f in hollow_feats)
h_rate = h_count / max(1, n_scored)
print(f"\nhollow-attributed: {h_count} over {n_scored} parts = {h_rate:.3f}/part")
print(f"v1 baseline rate: {V1_HOLLOW_PER_PART:.3f}/part (163/150)")
red = 1 - h_rate / V1_HOLLOW_PER_PART
print(f"reduction: {100*red:.1f}%  -> H1 {'CONFIRMED (>=50%)' if red >= 0.5 else 'NOT met (<50%)'}")
out = {"n_scored": n_scored, "n_hard": n_hard,
       "false_elims_all": total_fe_all, "false_elims_hard": total_fe_hard,
       "by_feature_all": dict(feat_all), "by_feature_hard": dict(feat_hard),
       "hollow_count": h_count, "hollow_per_part": round(h_rate, 4),
       "v1_hollow_per_part": round(V1_HOLLOW_PER_PART, 4),
       "reduction": round(red, 4), "h1_pass": bool(red >= 0.5)}
(MIP / "outputs" / "h1_tally_v2.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("wrote outputs/h1_tally_v2.json")
