# -*- coding: utf-8 -*-
"""Robustness pack (CC-3): two additive views of the frozen benchmark.

1. Verified-rules-only ablation: the five-way benchmark re-run with the 21
   clauses that were [VERIFY]-flagged before the CC-2 citation closure
   REMOVED (the pre-CC-2 71-clause verified subset). Answers hard question 9
   ("a quarter of your decision engine is unverified"): how much do the
   once-flagged clauses move the headline numbers?
2. Direct@k sweep, k in {5..10}: answers hard question 10 ("why k=8?") by
   showing whether the Hybrid-vs-Direct@k ordering is stable across k.

Both views reuse the frozen spec of outputs/benchmark_1000.json exactly
(size 1000, seeds 3, seed lineage 20260630, direct_topk 8, shortlist_k 3);
k and the clause subset are enumerated here, never selected post-hoc.

    python scripts/robustness_pack.py       # writes outputs/robustness.json
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

MIP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MIP / "src"))

from mip.benchmark import run_benchmark          # noqa: E402
from mip.config import Config                    # noqa: E402

RULES = MIP / "data" / "rules.json"
CANON = MIP / "outputs" / "benchmark_1000.json"
OUT = MIP / "outputs" / "robustness.json"

# The 21 clauses that carried "[VERIFY:" flags before the 2026-08-04 citation
# closure (CC-2). Snapshotted as tuples so this ablation stays reproducible
# after the flags were replaced by verified citations: the ablation subset is
# defined by WHICH clauses were once unverified, not by the marker.
FLAGGED_21 = {
    ("Injection_Molding", "continuous_fiber_required", "eq", True),
    ("Injection_Molding", "max_dimension_mm", "gt", 4000),
    ("Thermoforming", "max_dimension_mm", "gt", 6000),
    ("Compression_Molding", "max_dimension_mm", "gt", 4000),
    ("Vacuum_Assisted_RTM", "production_volume", "eq", 2),
    ("Structural_Reaction_Injection_Molding", "max_dimension_mm", "gt", 3500),
    ("Sheet_Molding_Compound", "max_dimension_mm", "gt", 4000),
    ("Bulk_Molding_Compound", "continuous_fiber_required", "eq", True),
    ("Bulk_Molding_Compound", "max_dimension_mm", "gt", 2000),
    ("Hand_Lay_up", "production_volume", "eq", 2),
    ("Blow_Molding", "continuous_fiber_required", "eq", True),
    ("Blow_Molding", "max_dimension_mm", "gt", 4000),
    ("Overmolding", "max_dimension_mm", "gt", 3000),
    ("Fused_Filament_Fabrication", "max_dimension_mm", "gt", 2200),
    ("Fused_Filament_Fabrication", "production_volume", "eq", 2),
    ("Stereolithography", "continuous_fiber_required", "eq", True),
    ("Stereolithography", "max_dimension_mm", "gt", 2100),
    ("Selective_Laser_Sintering", "continuous_fiber_required", "eq", True),
    ("Selective_Laser_Sintering", "max_dimension_mm", "gt", 1000),
    ("Laminated_Object_Manufacturing", "max_dimension_mm", "gt", 820),
    ("Laminated_Object_Manufacturing", "production_volume", "eq", 2),
}

KEEP = ("f1", "precision", "recall", "containment", "shortlist_f1",
        "avg_false_elims", "avg_set_size", "f1_ci95")
SWEEP_KS = [5, 6, 7, 8, 9, 10]


def _rows(summary: dict) -> dict:
    return {scope: {m: {k: r[k] for k in KEEP if k in r}
                    for m, r in summary[key].items()}
            for scope, key in (("all", "all_components"),
                               ("hard", "hard_subset"))}


def main() -> int:
    canon = json.loads(CANON.read_text(encoding="utf-8"))
    assert canon["direct_topk"] == 8 and canon["trials"] == 3

    raw = json.loads(RULES.read_text(encoding="utf-8"))
    removed = []
    for proc, body in raw["rules"].items():
        kept = []
        for c in body["forbidden_if"]:
            key = (proc, c["feature"], c["op"], c["value"])
            (removed if key in FLAGGED_21 else kept).append(c)
        body["forbidden_if"] = kept
    n_kept = sum(len(b["forbidden_if"]) for b in raw["rules"].values())
    if len(removed) != 21 or n_kept != 71:
        print(f"FATAL: expected to remove 21 of 92 clauses, removed "
              f"{len(removed)}, kept {n_kept} (rules.json drifted from the "
              f"FLAGGED_21 snapshot?)")
        return 1

    with tempfile.TemporaryDirectory() as td:
        alt = Path(td) / "rules_verified71.json"
        alt.write_text(json.dumps(raw), encoding="utf-8")

        print("== verified-71 ablation (size 1000, seeds 3, k=8) ==")
        abl = run_benchmark(Config(seeds=3), size=1000, n_boot=1000,
                            direct_topk=8, shortlist_k=3,
                            out_path=Path(td) / "abl.json", rules_path=alt)

        sweep = {}
        for k in SWEEP_KS:
            print(f"== Direct@{k} sweep run ==")
            s = run_benchmark(Config(seeds=3), size=1000, n_boot=1000,
                              direct_topk=k, shortlist_k=3,
                              out_path=Path(td) / f"k{k}.json")
            sweep[str(k)] = {
                scope: {m: {kk: s[key][m][kk] for kk in KEEP if kk in s[key][m]}
                        for m in ("Direct@k", "Hybrid")}
                for scope, key in (("all", "all_components"),
                                   ("hard", "hard_subset"))}

    abl_rows = _rows(abl)
    canon_rows = _rows(canon)
    deltas = {scope: {m: {"f1_delta": round(
                  abl_rows[scope][m]["f1"] - canon_rows[scope][m]["f1"], 6)}
              for m in canon_rows[scope]} for scope in ("all", "hard")}

    ordering = {k: sweep[k]["hard"]["Hybrid"]["f1"] >
                   sweep[k]["hard"]["Direct@k"]["f1"] for k in sweep}
    stable = all(ordering.values())
    sentence = ("The Hybrid-vs-Direct@k ordering on HARD is stable across the "
                "whole sweep: Hybrid macro-F1 exceeds Direct@k at every k in "
                "{5,6,7,8,9,10}." if stable else
                "The Hybrid-vs-Direct@k ordering on HARD is NOT stable across "
                f"the sweep: {ordering}")

    out = {
        "spec": {"size": 1000, "seeds": 3, "seed": 20260630, "n_boot": 1000,
                 "shortlist_k": 3, "canonical": "outputs/benchmark_1000.json",
                 "generator": "scripts/robustness_pack.py"},
        "verified_rules_only": {
            "n_clauses": abl["n_clauses"],
            "removed_clauses": 21,
            "rows": abl_rows,
            "f1_delta_vs_canonical": deltas,
        },
        "direct_k_sweep": {"ks": SWEEP_KS, "rows": sweep,
                           "hybrid_beats_direct_at_k_hard": ordering,
                           "stability_sentence": sentence},
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    print(sentence)
    for scope in ("all", "hard"):
        print(f"verified-71 F1 deltas ({scope.upper()}): " + "  ".join(
            f"{m} {deltas[scope][m]['f1_delta']:+.3f}" for m in deltas[scope]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
