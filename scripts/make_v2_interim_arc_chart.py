# -*- coding: utf-8 -*-
"""Real-photos slide figure: the bench-v2 read, interim-labeled.

Draws ONLY the in-progress preregistered run's disclosed interim readout
(outputs/benchmark_real_v2_interim.json): Hybrid macro F1 on ALL and HARD,
with two HARD reference marks read from committed artifacts, never retyped:
the H3 falsification line (.20, PREREGISTRATION_BENCH_V2.md) and the
clean-feature Hybrid ceiling (outputs/benchmark_1000.json). The v1 baseline
stays in the speaker notes and report 8.6. H1-H3 are judged SOLELY on the
completed closed pool; the figure states the slice size on its face.

Refuses to run if the interim artifact is missing, so the figure on the
slide always traces to a JSON in outputs/.

    python scripts/make_v2_interim_arc_chart.py
    # writes outputs/assets/real_v1_arc_interim.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MIP = Path(__file__).resolve().parent.parent
OUT = MIP / "outputs"

# frozen constant from PREREGISTRATION_BENCH_V2.md (do not edit)
H3_LINE = 0.20
POOL = 635

NAVY, CORAL, INK, MUTED = "#00539F", "#E8663C", "#22262B", "#6B7280"
CORAL_LT = "#F6CDBB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interim", default=str(OUT / "benchmark_real_v2_interim.json"))
    ap.add_argument("--b1000", default=str(OUT / "benchmark_1000.json"))
    ap.add_argument("--out", default=str(OUT / "assets" / "real_v1_arc_interim.png"))
    a = ap.parse_args()

    ip = Path(a.interim)
    if not ip.exists():
        print("REFUSING: interim artifact not found: " + str(ip) + ". The slide "
              "figure only draws numbers that trace to a JSON in outputs/. Run "
              "scripts/score_v2_interim.py once the in-progress run's cache "
              "holds a scoreable stratified slice, then rerun this script.")
        return 2
    iv = json.loads(ip.read_text(encoding="utf-8"))
    if not iv.get("INTERIM"):
        print("REFUSING: " + str(ip) + " is not marked INTERIM; for the "
              "completed-pool figure use make_v2_arc_chart.py instead.")
        return 2
    ceiling = json.loads(Path(a.b1000).read_text(encoding="utf-8"))[
        "hard_subset"]["Hybrid"]["f1"]

    f_all = iv["all_components"]["Hybrid"]["f1"]
    f_hard = iv["hard_subset"]["Hybrid"]["f1"]
    ci_a = iv["all_components"]["Hybrid"].get("f1_ci95")
    ci_h = iv["hard_subset"]["Hybrid"].get("f1_ci95")
    n_i = iv.get("dataset_size")
    n_txt = "n=" + (str(n_i) if n_i is not None else "?")

    fig, ax = plt.subplots(figsize=(4.45, 2.72), dpi=300)
    fig.patch.set_facecolor("white")

    vals = [f_all, f_hard]
    cis = [ci_a, ci_h]
    ax.bar([0, 1], vals, width=0.42, color=[CORAL_LT, CORAL], zorder=3,
           hatch=[None, "///"], edgecolor=["none", "white"], linewidth=0,
           yerr=[[0 if not c else v - c[0] for v, c in zip(vals, cis)],
                 [0 if not c else c[1] - v for v, c in zip(vals, cis)]],
           error_kw=dict(ecolor=INK, lw=0.9, capsize=2.5))
    for x, v, c in zip([0, 1], vals, cis):
        top = v if not c else c[1]
        ax.text(x, top + 0.014, ("%.3f" % v).lstrip("0"), ha="center",
                fontsize=8.8, fontweight="bold", color=INK)

    # HARD-only reference marks, drawn over the HARD bar region
    ax.plot([0.62, 1.42], [H3_LINE, H3_LINE], color=NAVY, lw=1.1,
            ls=(0, (5, 4)), zorder=2)
    ax.text(0.60, H3_LINE - 0.012, "H3 falsification line (.20,\ncommitted before the run)",
            ha="right", va="center", fontsize=5.9, color=NAVY, style="italic")
    ax.plot([0.62, 1.42], [ceiling, ceiling], color=MUTED, lw=1.0,
            ls=(0, (1, 2)), zorder=2)
    ax.text(0.60, ceiling, "clean-feature Hybrid ceiling ("
            + ("%.3f" % ceiling).lstrip("0") + " HARD)",
            ha="right", va="center", fontsize=5.9, color=MUTED, style="italic")

    ax.set_title("bench-v2, the run in flight: Hybrid macro F1 (interim, "
                 + n_txt + " of " + str(POOL) + ")", fontsize=8.0, color=INK, pad=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["ALL", "HARD (thesis subset)"], fontsize=7.2, color=INK)
    ax.set_ylim(0, max(0.44, ceiling + 0.05, f_hard + 0.06))
    ax.set_ylabel("macro F1", fontsize=7.2, color=INK)
    cap = ("stratified slice, frozen pipeline; H1-H3 are judged only on the "
           "completed " + str(POOL) + "-photo pool and ship either way.")
    ax.text(0.0, -0.155, cap, fontsize=5.8, color=MUTED, style="italic",
            va="top", ha="left", transform=ax.transAxes)

    ax.tick_params(length=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#D5DBE3")
    ax.spines["bottom"].set_color("#D5DBE3")
    ax.grid(axis="y", color="#EDF1F5", lw=0.7, zorder=0)
    ax.tick_params(axis="y", labelsize=6.6, colors=INK)

    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outp, facecolor="white", bbox_inches="tight", pad_inches=0.06)
    print("wrote " + str(outp))
    print("interim Hybrid ALL %.3f / HARD %.3f; H3 %.2f; ceiling %.3f; n=%s"
          % (f_all, f_hard, H3_LINE, ceiling, n_i))
    return 0


if __name__ == "__main__":
    sys.exit(main())
