# -*- coding: utf-8 -*-
"""CC-1 replacement figures, regenerated from committed JSONs + run cache.

Figure A (Observation slide): per-process share of picks, model vs experts.
  Sources: outputs/observation_stats.json (model, 150 real photos) and
  outputs/survey_stats.json (experts, 42 survey parts, 427 picks). The two
  populations differ, so shares are compared and the figure says so.

Figure B (reframed Retrieval slide -> "the rule engine moves the decision"):
  per-instance PAIRED comparison on the same 150 photos of the archived
  benchmark_real run: size-matched Direct@8 containment vs Hybrid survivor
  containment, with McNemar's exact test on the discordant pairs printed on
  the figure. Sources: outputs/hybrid_graphs.json (Hybrid survivors + expert
  sets) and the run cache's top-8 responses (local-only, like
  observation_stats.py).

    python scripts/make_survey_figures.py   # writes outputs/assets/*.png
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MIP = Path(__file__).resolve().parent.parent
OUT = MIP / "outputs"
ASSETS = OUT / "assets"
CACHE = OUT / "vlm_cache" / "bench-gemini-2.5-flash-lite"
POOL = MIP / "images_pool"

NAVY = "#1f3864"
CORAL = "#ff6f61"
GREY = "#9aa3ad"
TEAL = "#2a9d8f"

random.seed(20260630)  # house seed lineage; no randomness is drawn today

MERGED = {"Compression_Molding": "Compression_Molding_or_Thermoforming",
          "Thermoforming": "Compression_Molding_or_Thermoforming"}

PRETTY = {
    "Injection_Molding": "Injection Molding",
    "Compression_Molding_or_Thermoforming": "Compr. Mold. / Thermof.",
    "Resin_Transfer_Molding": "RTM",
    "Sheet_Molding_Compound": "SMC",
    "Vacuum_Assisted_RTM": "VARTM",
    "Structural_Reaction_Injection_Molding": "SRIM",
    "Bulk_Molding_Compound": "BMC",
    "Hand_Lay_up": "Hand Lay-up",
    "Selective_Laser_Sintering": "SLS",
    "Stereolithography": "SLA",
    "Pultrusion": "Pultrusion",
    "Laminated_Object_Manufacturing": "LOM",
    "Fused_Filament_Fabrication": "FFF",
    "Filament_Winding": "Filament Winding",
    "Blow_Molding": "Blow Molding",
    "Overmolding": "Overmolding",
    "Adhesive_Bonding": "Adhesive Bonding (survey only)",
    "NA_none_of_the_listed": "N/A (survey only)",
}


def _load(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def figure_a() -> Path:
    """Presentation-grade restyle (2026-08-04): same numbers, same sources.

    Processes at >=3% share on either side get their own row; the rest are
    aggregated into one labeled remainder row (full distribution lives in
    survey_stats.json / observation_stats.json and report 8.1).
    """
    NAVY2, CORAL2, INK, MUTED = "#00539F", "#E8663C", "#22262B", "#6B7280"
    obs = _load("observation_stats.json")
    survey = _load("survey_stats.json")
    model = obs["model"]["totals"]
    expert = survey["expert_totals"]
    n_m = sum(model.values())
    n_e = sum(expert.values())

    all_procs = set(model) | set(expert)
    big = [p for p in all_procs
           if model.get(p, 0) / n_m >= 0.03 or expert.get(p, 0) / n_e >= 0.03]
    small = sorted(all_procs - set(big))
    order = ["Injection_Molding"] + sorted(
        [p for p in big if p != "Injection_Molding"],
        key=lambda p: -expert.get(p, 0))
    rows = [(PRETTY.get(p, p), model.get(p, 0) / n_m, expert.get(p, 0) / n_e)
            for p in order]
    rows.append((f"{len(small)} remaining options",
                 sum(model.get(p, 0) for p in small) / n_m,
                 sum(expert.get(p, 0) for p in small) / n_e))

    fig, ax = plt.subplots(figsize=(7.0, 6.2), dpi=300)
    fig.patch.set_facecolor("white")
    ys = range(len(rows))
    h = 0.40
    m_share = [r[1] for r in rows]
    e_share = [r[2] for r in rows]
    ax.barh([y - h / 2 for y in ys], m_share, height=h, color=CORAL2, zorder=3)
    ax.barh([y + h / 2 for y in ys], e_share, height=h, color=NAVY2, zorder=3)
    for y, v in zip(ys, m_share):
        if v >= 0.005:
            ax.text(v + 0.012, y - h / 2, f"{v:.0%}", va="center",
                    fontsize=13 if v > 0.5 else 10.5, color=CORAL2,
                    fontweight="bold" if v > 0.5 else "normal")
    for y, v in zip(ys, e_share):
        if v >= 0.005:
            ax.text(v + 0.012, y + h / 2, f"{v:.0%}", va="center",
                    fontsize=10.5, color=NAVY2)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows], fontsize=11.5, color=INK)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.get_yticklabels()[-1].set_color(MUTED)
    ax.get_yticklabels()[-1].set_fontsize(10)
    ax.invert_yaxis()
    ax.set_xlim(0, 0.86)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%"], fontsize=10,
                       color=MUTED)
    ax.tick_params(length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#D5DBE3")
    ax.text(0.82, len(rows) - 2.35, f"model (1 pick / photo, n={n_m})",
            ha="right", fontsize=11.5, color=CORAL2, fontweight="bold")
    ax.text(0.82, len(rows) - 1.95, f"experts (multi-select, n={n_e} picks)",
            ha="right", fontsize=11.5, color=NAVY2, fontweight="bold")
    ax.set_ylim(len(rows) - 0.5, -0.6)
    fig.subplots_adjust(left=0.315, right=0.985, top=0.995, bottom=0.055)
    p = ASSETS / "observation_model_vs_expert.png"
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    return p


def _sha24(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:24]


def _find_image(instance: str) -> Path:
    cat = instance.rsplit("_", 1)[0]
    for base in (POOL / cat, POOL / "_rejected" / cat, POOL / "_rejected"):
        cand = base / f"{instance}.jpg"
        if cand.exists():
            return cand
    raise FileNotFoundError(instance)


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact binomial test on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def figure_b() -> Path:
    graphs = _load("hybrid_graphs.json")
    n11 = n10 = n01 = n00 = 0
    d_hits = h_hits = 0
    for g in graphs:
        expert = set(g["expert"])
        img = _find_image(g["instance"])
        sha = _sha24(img)
        topk_file = CACHE / f"{sha}_topk-__process_top8___00_bench-v1.json"
        raw = json.loads(topk_file.read_text(encoding="utf-8"))["raw"]
        top8 = [t.strip() for t in raw.splitlines() if t.strip()]
        d = expert <= set(top8)
        h = expert <= set(g["graph"]["survivors"])
        d_hits += d
        h_hits += h
        n11 += d and h
        n10 += d and not h
        n01 += h and not d
        n00 += (not d) and (not h)

    n = len(graphs)
    p = mcnemar_exact_p(n10, n01)

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    bars = ax.bar([0, 1], [d_hits / n, h_hits / n], width=0.5,
                  color=[GREY, CORAL])
    ax.set_xticks([0, 1], [f"Direct@8 (size-matched ranking)\n"
                           f"{d_hits}/{n} parts contain the expert set",
                           f"Hybrid (perceive + rule engine)\n"
                           f"{h_hits}/{n} parts contain the expert set"],
                  fontsize=9)
    for b_, v in zip(bars, (d_hits / n, h_hits / n)):
        ax.text(b_.get_x() + b_.get_width() / 2, b_.get_height() + 0.02,
                f"{v:.1%}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("full expert-set containment (binary, per instance)",
                  fontsize=9)
    ax.set_title("The rule engine is what changes the decision - and with "
                 "v1 real-photo perception\nit changes it for the worse: "
                 f"{n10} flips against vs {n01} for "
                 f"(McNemar exact p = {p:.1e})", fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = ASSETS / "decision_direct_vs_hybrid.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)

    (OUT / "decision_paired_stats.json").write_text(json.dumps({
        "source": {
            "pairing": "per-instance, benchmark_real run (150 photos)",
            "direct": "run cache top-8 responses (local-only)",
            "hybrid": "outputs/hybrid_graphs.json survivors",
            "generator": "scripts/make_survey_figures.py",
        },
        "n": n,
        "containment": {"direct_at_8": d_hits, "hybrid": h_hits},
        "paired_cells": {"both": n11, "direct_only": n10,
                         "hybrid_only": n01, "neither": n00},
        "mcnemar_exact_p": p,
        "metric_note": "Binary full-set containment (expert subseteq "
                       "decision set) per instance, sample-00 responses of "
                       "the archived run. This is stricter than the "
                       "benchmark's fractional containment metric (real "
                       "Direct@8 .835), so the two numbers coexist and "
                       "must not be conflated.",
        "claim_supported": "On the same 150 photos the decision-moving "
                           "component is the rule engine, not the ranker: "
                           f"{n10} paired flips against containment vs "
                           f"{n01} for. With v1 (pre-bench-v2) perception "
                           "the rules eliminate true expert processes, "
                           "which is the measured collapse (Hybrid HARD "
                           "F1 .086) and the case for calibrated "
                           "perception first. No retrieval arm exists in "
                           "the repo; the old with/without-retrieval "
                           "agreement chart was assertion-only and is "
                           "replaced by this paired comparison.",
    }, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    a = figure_a()
    print(f"wrote {a}")
    b = figure_b()
    print(f"wrote {b}")
    print(f"wrote {OUT / 'decision_paired_stats.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
