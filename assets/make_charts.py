"""Generate the results charts (goal 11, CLAUDE_CODE_GOALS.md) from the
benchmark output JSONs. Every plotted value is read straight from an
outputs/*.json produced by a command logged in REPRODUCIBILITY.md and is
annotated on the chart to three decimals, so chart-vs-JSON consistency is
checkable by eye.

    python assets/make_charts.py            # writes PNGs next to this script

Charts:
  ablation_all_hard.png     F1 by method, ALL vs HARD, 95% bootstrap CI whiskers
  survivor_containment.png  survivor-set narrowing vs expert containment
  real_robustness.png       mock vs real backend (only if benchmark_real.json exists)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent          # mip/assets
OUT = HERE.parent / "outputs"

NAVY = "#1f3864"
CORAL = "#ff6f61"
GREY = "#9aa3ad"
TEAL = "#2a9d8f"
GOLD = "#e9c46a"

METHOD_COLORS = {"Direct": GREY, "RuleSet": NAVY, "Greedy": GOLD,
                 "Hybrid": CORAL, "Direct@k": "#6c757d"}


def _load(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def _bar_with_ci(ax, rows: dict, methods, title: str):
    xs = range(len(methods))
    f1 = [rows[m]["f1"] for m in methods]
    lo = [rows[m]["f1"] - rows[m]["f1_ci95"][0] for m in methods]
    hi = [rows[m]["f1_ci95"][1] - rows[m]["f1"] for m in methods]
    bars = ax.bar(xs, f1, color=[METHOD_COLORS.get(m, GREY) for m in methods],
                  yerr=[lo, hi], capsize=4, error_kw={"lw": 1.2})
    for b, v in zip(bars, f1):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.025,
                f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(list(xs), methods, fontsize=9)
    ax.set_ylim(0, 0.75)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("macro F1")
    ax.spines[["top", "right"]].set_visible(False)


def chart_ablation(summary: dict, fname="ablation_all_hard.png"):
    methods = list(summary["all_components"].keys())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    _bar_with_ci(axes[0], summary["all_components"], methods,
                 "ALL components (IM-heavy, bias often looks right)")
    _bar_with_ci(axes[1], summary["hard_subset"], methods,
                 "HARD subset (expert set is not just injection molding)")
    n = summary["dataset_size"]
    t = summary["trials"]
    fig.suptitle(f"Three-way ablation + set-size-matched Direct@k  "
                 f"(MockVLM, {n} instances, {t} trials, seed "
                 f"{summary['config']['seed']}, 95% bootstrap CI)", fontsize=11)
    fig.tight_layout()
    fig.savefig(HERE / fname, dpi=200)
    plt.close(fig)
    print(f"wrote {HERE / fname}")


def chart_survivors(summary: dict, fname="survivor_containment.png"):
    methods = [m for m in summary["all_components"] if m != "Direct"]
    rows_a = summary["all_components"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    xs = range(len(methods))
    sizes = [rows_a[m]["avg_set_size"] for m in methods]
    bars = ax.bar(xs, sizes, color=[METHOD_COLORS.get(m, GREY) for m in methods])
    ax.axhline(17, color="black", lw=1, ls="--")
    ax.text(len(methods) - 0.5, 17.3, "17 candidate processes", fontsize=8,
            ha="right")
    for b, v in zip(bars, sizes):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(list(xs), methods, fontsize=9)
    ax.set_ylim(0, 19)
    ax.set_title("Mean survivor-set size (how far 17 was narrowed)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    cont = [rows_a[m]["containment"] for m in methods]
    bars = ax.bar(xs, cont, color=[METHOD_COLORS.get(m, GREY) for m in methods])
    for b, v in zip(bars, cont):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(list(xs), methods, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_title("Expert-containment rate (full expert set survived)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Elimination-native view: safe narrowing, not just set-F1 "
                 f"(ALL components, {summary['trials']} trials)", fontsize=11)
    fig.tight_layout()
    fig.savefig(HERE / fname, dpi=200)
    plt.close(fig)
    print(f"wrote {HERE / fname}")


def chart_real(mock: dict, real: dict, fname="real_robustness.png"):
    methods = [m for m in real["all_components"] if m in mock["all_components"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, block, title in ((axes[0], "all_components", "ALL"),
                             (axes[1], "hard_subset", "HARD subset")):
        xs = range(len(methods))
        w = 0.38
        mv = [mock[block][m]["f1"] for m in methods]
        rv = [real[block][m]["f1"] for m in methods]
        b1 = ax.bar([x - w / 2 for x in xs], mv, w, color=GREY,
                    label=f"MockVLM ({mock['dataset_size']} inst)")
        b2 = ax.bar([x + w / 2 for x in xs], rv, w, color=TEAL,
                    label=f"{real['backend']} on {real['dataset_size']} real photos")
        for bars, vals in ((b1, mv), (b2, rv)):
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                        f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")
        ax.set_xticks(list(xs), methods, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("macro F1")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8)
    fig.suptitle("Robustness: the ablation ordering holds on real CC-licensed "
                 "photos (category-level expert labels)", fontsize=11)
    fig.tight_layout()
    fig.savefig(HERE / fname, dpi=200)
    plt.close(fig)
    print(f"wrote {HERE / fname}")


def main() -> int:
    summary = _load("benchmark_summary.json")
    chart_ablation(summary)
    chart_survivors(summary)
    real_p = OUT / "benchmark_real.json"
    if real_p.exists():
        chart_real(summary, _load("benchmark_real.json"))
    else:
        print("benchmark_real.json not present yet; skipping real_robustness.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
