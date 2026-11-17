# -*- coding: utf-8 -*-
"""Before/after arc chart for the real-photos slide, built ONLY when the
preregistered bench-v2 full-pool run is complete.

HARD GATE: refuses to run unless outputs/benchmark_real_v2.json exists and
covers the closed pool. No partial-cache scoring, no interim peeks; the
preregistered analysis is one pass on the completed pool (PREREGISTRATION_
BENCH_V2.md, frozen at c84fbc3). H2/H3 verdicts are computed here from the
frozen constants and printed for RESULTS_CC4.md.

    python scripts/make_v2_arc_chart.py   # writes outputs/assets/v2_before_after.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MIP = Path(__file__).resolve().parent.parent
OUT = MIP / "outputs"

# frozen constants from PREREGISTRATION_BENCH_V2.md (do not edit)
H3_LINE = 0.20
V1_HYBRID_HARD = 0.086
POOL_MIN = 600  # closed pool is 635; refuse anything materially smaller

NAVY, CORAL, INK, MUTED = "#00539F", "#E8663C", "#22262B", "#6B7280"
CORAL_LT, NAVY_LT = "#F6CDBB", "#C9D6E8"


def main() -> int:
    v2_path = OUT / "benchmark_real_v2.json"
    if not v2_path.exists():
        print("REFUSING: outputs/benchmark_real_v2.json does not exist yet. "
              "The preregistered analysis is one scoring pass on the COMPLETED "
              "closed pool; partial-cache numbers are not results. Let the run "
              "finish, then rerun this script.")
        return 2
    v1 = json.loads((OUT / "benchmark_real.json").read_text(encoding="utf-8"))
    v2 = json.loads(v2_path.read_text(encoding="utf-8"))
    if v2.get("dataset") != "real" or int(v2.get("dataset_size", 0)) < POOL_MIN:
        print(f"REFUSING: v2 file is not a full closed-pool real run "
              f"(dataset={v2.get('dataset')}, n={v2.get('dataset_size')}).")
        return 2

    v1h, v2h = v1["hard_subset"], v2["hard_subset"]
    assert abs(v1h["Hybrid"]["f1"] - V1_HYBRID_HARD) < 1e-9, "v1 baseline moved?"
    hyb2 = v2h["Hybrid"]["f1"]
    ci = v2h["Hybrid"].get("f1_ci95")
    h2 = hyb2 > V1_HYBRID_HARD
    h3_crossed = hyb2 >= H3_LINE

    labels = ["v1 run\n(150 photos,\nold phrasing)", "v2 run\n(closed pool, "
              f"n={v2['dataset_size']},\npreregistered fix)"]
    vals = [V1_HYBRID_HARD, hyb2]
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=300)
    fig.patch.set_facecolor("white")
    bars = ax.bar([0, 1], vals, width=0.5, color=[CORAL_LT, CORAL], zorder=3,
                  yerr=[[0, 0 if not ci else hyb2 - ci[0]],
                        [0, 0 if not ci else ci[1] - hyb2]],
                  error_kw=dict(ecolor=INK, lw=1.2, capsize=4))
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012,
                f"{v:.3f}".lstrip("0"), ha="center", fontsize=13,
                fontweight="bold", color=INK)
    ax.axhline(H3_LINE, color=NAVY, lw=1.4, ls=(0, (5, 4)), zorder=2)
    ax.text(1.38, H3_LINE + 0.006,
            "H3 falsification line (.20, committed before the run)",
            ha="right", fontsize=10, color=NAVY, style="italic")
    verdict = (f"H2 {'CONFIRMED' if h2 else 'REFUTED'}: "
               f"{hyb2:.3f} vs v1 {V1_HYBRID_HARD:.3f}   |   "
               f"H3 line {'CROSSED' if h3_crossed else 'NOT crossed'}"
               f"{'' if h3_crossed else ' (calibration-first stands, as preregistered)'}")
    ax.set_title(verdict, fontsize=11, color=INK, pad=10)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels, fontsize=11, color=INK)
    ax.set_ylabel("Hybrid macro F1, HARD subset", fontsize=11, color=INK)
    ax.set_ylim(0, max(0.42, hyb2 + 0.08))
    ax.tick_params(length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#D5DBE3")
    ax.spines["bottom"].set_color("#D5DBE3")
    ax.grid(axis="y", color="#EDF1F5", lw=0.8, zorder=0)

    outp = OUT / "assets" / "v2_before_after.png"
    outp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outp, facecolor="white")
    print(f"wrote {outp}")
    print(f"VERDICTS: H2={'CONFIRMED' if h2 else 'REFUTED'} "
          f"(Hybrid HARD {hyb2:.3f} vs {V1_HYBRID_HARD:.3f}); "
          f"H3 {'crossed' if h3_crossed else 'not crossed'} at {H3_LINE:.2f}. "
          f"H1 (hollow false-elim reduction) needs the v2 provenance tallies; "
          f"see CC-4 DO#3.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
