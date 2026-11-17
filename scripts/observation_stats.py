# -*- coding: utf-8 -*-
"""Model-side observation tally (CC-1): what the VLM actually picked.

The deck's original claim (injection molding 217 vs 57 over 280 images) has
no surviving artifact and exceeds the survey's arithmetic maximum, so it is
unreproducible. The honest replacement: per-image Direct picks from the
archived benchmark_real.json run (gemini-2.5-flash-lite, 150 stratified real
photos, REPRODUCIBILITY.md), read from the run's response cache.

The cache is local-only (gitignored); the committed artifact is
outputs/observation_stats.json. Re-running requires the cache of the archived
run to be present, and the script fails loudly if the tally cannot reproduce
the exact 150-image census of that run.

    python scripts/observation_stats.py   # writes outputs/observation_stats.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

MIP = Path(__file__).resolve().parent.parent
CACHE = MIP / "outputs" / "vlm_cache" / "bench-gemini-2.5-flash-lite"
SURVEY = MIP / "outputs" / "survey_stats.json"
OUT = MIP / "outputs" / "observation_stats.json"

# survey vocabulary merges these two processes into one option; the model
# side is folded into the same bucket so the two columns are comparable
MERGED = {"Compression_Molding": "Compression_Molding_or_Thermoforming",
          "Thermoforming": "Compression_Molding_or_Thermoforming"}


def main() -> int:
    files = sorted(CACHE.glob("*_direct-__process___00_bench-v1.json"))
    if not files:
        print(f"FATAL: no direct-pick cache files under {CACHE}; the "
              f"archived benchmark_real run's cache is required")
        return 1

    picks = Counter()
    for f in files:
        raw = json.loads(f.read_text(encoding="utf-8"))["raw"].strip()
        picks[MERGED.get(raw, raw)] += 1
    n = sum(picks.values())
    if n != 150:
        print(f"FATAL: expected the archived run's 150 direct picks, "
              f"found {n} (cache incomplete or polluted by another run?)")
        return 1

    survey = json.loads(SURVEY.read_text(encoding="utf-8"))
    expert = survey["expert_totals"]
    e_total = sum(expert.values())
    ranked_m = picks.most_common()
    ranked_e = sorted(expert.items(), key=lambda kv: (-kv[1], kv[0]))
    top5_m = [p for p, _ in ranked_m[:5]]
    top5_e = [p for p, _ in ranked_e[:5]]

    out = {
        "source": {
            "run": "outputs/benchmark_real.json (gemini-2.5-flash-lite, "
                   "150 stratified real photos)",
            "cache": "outputs/vlm_cache/bench-gemini-2.5-flash-lite "
                     "(*_direct-__process___00_bench-v1.json, local-only)",
            "generator": "scripts/observation_stats.py",
            "replaces": "the unreproducible '217 vs 57 over 280 images' "
                        "claim: no artifact of that experiment exists, and "
                        "217 exceeds the survey's arithmetic maximum of 210",
        },
        "populations_note": (
            "Model picks cover the 150-photo stratified benchmark pool; "
            "expert picks cover the 42 survey parts. Different image "
            "populations: the figure compares pick SHARES per process and "
            "says so on its face; it is not a per-part paired comparison."),
        "model": {
            "n_images": n,
            "totals": dict(sorted(picks.items())),
            "top_process": ranked_m[0][0],
            "top_share": round(ranked_m[0][1] / n, 3),
            "top5": top5_m,
        },
        "experts": {
            "n_picks": e_total,
            "top_process": ranked_e[0][0],
            "top_share": round(ranked_e[0][1] / e_total, 3),
            "top5": top5_e,
        },
        "top5_overlap": {
            "processes": sorted(set(top5_m) & set(top5_e)),
            "count": len(set(top5_m) & set(top5_e)),
            "replaces": "the 'top five overlap experts on only one process' "
                        "claim",
        },
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"model: {ranked_m[0][0]} {ranked_m[0][1]}/{n} "
          f"({out['model']['top_share']:.1%})")
    print(f"experts: {ranked_e[0][0]} {ranked_e[0][1]}/{e_total} "
          f"({out['experts']['top_share']:.1%})")
    print(f"top5 overlap: {out['top5_overlap']['count']} "
          f"({', '.join(out['top5_overlap']['processes'])})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
