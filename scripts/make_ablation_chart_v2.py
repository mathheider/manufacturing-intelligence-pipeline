# -*- coding: utf-8 -*-
"""Presentation-grade five-way ablation chart for deck s26.

Same numbers as the archived canonical run (outputs/benchmark_1000.json); style
only. In-bar percentages = share of each subset ceiling (RuleSet with oracle
features); the Direct family is a different decision logic, not bounded by it.
Deterministic: python scripts/make_ablation_chart_v2.py
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MIP = Path(__file__).resolve().parent.parent
B = json.loads((MIP / "outputs" / "benchmark_1000.json").read_text(encoding="utf-8"))
ab, hb = B["all_components"], B["hard_subset"]

NAVY, CORAL, INK, MUTED = "#00539F", "#E8663C", "#22262B", "#6B7280"
ALL_N, HARD_N = "#C9D6E8", "#4C7BB0"       # neutral pipelines
ALL_H, HARD_H = "#F6CDBB", "#E8663C"       # hybrid

ORDER = [("Direct", "Direct\n(singleton)"),
         ("Direct@k", "Direct@8\n(size-matched)"),
         ("RuleSet", "Rule Set\n(oracle features)"),
         ("Greedy", "Greedy\n(no consistency)"),
         ("Hybrid", "Hybrid\n(the MIP)")]

fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=300)
fig.patch.set_facecolor("white")
w = 0.36
for i, (k, label) in enumerate(ORDER):
    hyb = k == "Hybrid"
    ca, ch = (ALL_H, HARD_H) if hyb else (ALL_N, HARD_N)
    va, vh = ab[k]["f1"], hb[k]["f1"]
    ea = ab[k].get("f1_ci95"); eh = hb[k].get("f1_ci95")
    ax.bar(i - w / 2, va, width=w, color=ca, zorder=3,
           yerr=None if not ea else [[va - ea[0]], [ea[1] - va]],
           error_kw=dict(ecolor=MUTED, lw=1.2, capsize=3))
    ax.bar(i + w / 2, vh, width=w, color=ch, zorder=3,
           yerr=None if not eh else [[vh - eh[0]], [eh[1] - vh]],
           error_kw=dict(ecolor=INK, lw=1.2, capsize=3))
    ax.text(i - w / 2, va + 0.035, f"{va:.3f}".lstrip("0"), ha="center",
            fontsize=11, color=MUTED)
    ax.text(i + w / 2, vh + 0.035, f"{vh:.3f}".lstrip("0"), ha="center",
            fontsize=12.5, color=INK,
            fontweight="bold" if hyb or k == "RuleSet" else "normal")

# ceiling annotation
ceil = hb["RuleSet"]["f1"]
ax.axhline(ceil, xmin=0.38, xmax=1.0, color=NAVY, lw=1.4, ls=(0, (5, 4)), zorder=2)
ax.text(4.42, ceil + 0.014, "perfect-perception ceiling (.510 HARD); in-bar % = share of each subset's ceiling",
        ha="right", fontsize=10.5, color=NAVY, style="italic")
hyb_h = hb["Hybrid"]["f1"]
# normalized read, drawn in the bars: % of the perfect-perception ceiling,
# per subset (ALL ceiling .375, HARD ceiling .510). The Direct family is a
# different decision logic and is not bounded by the ceiling, so no % there.
ceil_all = ab["RuleSet"]["f1"]
for i, (k, _) in enumerate(ORDER):
    if k not in ("RuleSet", "Greedy", "Hybrid"):
        continue
    for dx, v, cl, light in ((-w / 2, ab[k]["f1"], ceil_all, True),
                             (w / 2, hb[k]["f1"], ceil, False)):
        pct = "= ceiling" if k == "RuleSet" else f"{v / cl:.0%}"
        ax.text(i + dx, v - 0.016, pct, ha="center", va="top",
                fontsize=8.6 if k == "RuleSet" else 9.5,
                color="#5B6B7C" if light else "white",
                fontweight="bold", rotation=90 if k == "RuleSet" else 0)


ax.set_xticks(range(5))
ax.set_xticklabels([l for _, l in ORDER], fontsize=11.5, color=INK)
ax.set_ylim(0, 0.64)
ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
ax.set_yticklabels(["0", ".1", ".2", ".3", ".4", ".5", ".6"], fontsize=10, color=MUTED)
ax.set_ylabel("macro F1 (95% bootstrap CI)", fontsize=11, color=INK)
ax.tick_params(length=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color("#D5DBE3"); ax.spines["bottom"].set_color("#D5DBE3")
ax.grid(axis="y", color="#EDF1F5", lw=0.8, zorder=0)
# legend
from matplotlib.patches import Patch
ax.legend(handles=[Patch(fc=ALL_N, label="ALL (1,000 instances)"),
                   Patch(fc=HARD_N, label="HARD (expert set beyond IM)")],
          loc="upper left", fontsize=10.5, frameon=False)
ax.set_title("Five-way ablation, 1,000 instances, 3 trials, seed 20260630",
             fontsize=11, color=MUTED, pad=10)
fig.subplots_adjust(left=0.085, right=0.99, top=0.93, bottom=0.11)
OUTP = MIP / "outputs" / "assets" / "ablation_fiveway_v2.png"
OUTP.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTP, facecolor="white")
print(f"wrote {OUTP} ceiling", ceil, "hybrid", hyb_h)
