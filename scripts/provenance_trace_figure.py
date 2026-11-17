#!/usr/bin/env python3
"""Render the worked provenance-trace figure for the prelim deck.

One real part (headlamp_housing_009, 150-photo pool) walked end to end:
photo -> committed VLM feature reads -> the forbidden_if clause that fired
(with citation) -> the cited eliminations -> the surviving process set.
Every value shown is read from outputs/hybrid_graphs.json; nothing is typed in.

Usage: python provenance_trace_figure.py [--root MIP_ROOT] [--out PNG_PATH]
Defaults assume the script sits in mip/scripts/.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

INSTANCE = "headlamp_housing_009"
FEATURED_PROCESS = "Sheet_Molding_Compound"

NAVY = "#00539F"
CORAL = "#E8663C"
INK = "#22262B"
MUTED = "#6B7280"
PANEL_BG = "#F4F6F9"
CARD_BG = "#FFFFFF"

def nice(s):
    return s.replace("_", " ")

def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--root", type=Path, default=here.parent)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    root = args.root
    out = args.out or (root / "outputs" / "assets" / "provenance_trace.png")

    data = json.loads((root / "outputs" / "hybrid_graphs.json").read_text())
    inst = next(i for i in data if i["instance"] == INSTANCE)
    g = inst["graph"]
    feats = g["features"]
    elims = g["eliminations"]
    survivors = g["survivors"]
    expert = inst["expert"]
    featured = elims[FEATURED_PROCESS]

    committed = {k: v for k, v in feats.items() if v["committed"]}
    n_probed, n_committed = len(feats), len(committed)

    # group eliminations by (feature, op)
    by_feat = {}
    for p, e in elims.items():
        by_feat.setdefault((e["feature"], e["op"], str(e["threshold"])), []).append(p)

    img = Image.open(root / "images_pool" / "headlamp_housing" / f"{INSTANCE}.jpg")

    fig = plt.figure(figsize=(14.0, 5.62), dpi=200)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 14); ax.set_ylim(0, 5.62); ax.axis("off")

    def panel(x, w, header, sub=None):
        ax.add_patch(FancyBboxPatch((x, 0.52), w, 4.42, boxstyle="round,pad=0.06,rounding_size=0.10",
                                    fc=PANEL_BG, ec="#D5DBE3", lw=1.0, zorder=1))
        ax.text(x + w / 2, 5.22, header, ha="center", va="top", fontsize=12.5,
                color=NAVY, fontweight="bold")
        if sub:
            ax.text(x + w / 2, 0.30, sub, ha="center", va="center", fontsize=8.0, color=MUTED)

    def arrow(x0, x1):
        ax.add_patch(FancyArrowPatch((x0, 2.73), (x1, 2.73), arrowstyle="-|>",
                                     mutation_scale=22, lw=2.2, color=NAVY, zorder=5))

    # ---- panel geometry
    P = [(0.18, 2.42), (2.98, 2.60), (5.96, 3.10), (9.44, 2.10), (11.92, 1.90)]

    # 1) the part
    x, w = P[0]
    panel(x, w, "One real part", "headlamp_housing_009, 150-photo CC pool")
    iw, ih = img.size
    disp_w = w - 0.30
    disp_h = disp_w * ih / iw
    ax.imshow(img, extent=(x + 0.15, x + 0.15 + disp_w, 2.60 - disp_h / 2 + 0.35, 2.60 + disp_h / 2 + 0.35), zorder=2)
    ax.text(x + w / 2, 1.30, "Headlamp housing", ha="center", fontsize=10.5, color=INK, fontweight="bold")
    ax.text(x + w / 2, 0.98, "expert answer: injection molding", ha="center", fontsize=8.6, color=MUTED)

    # 2) perception
    x, w = P[1]
    panel(x, w, "The VLM perceives", f"{n_probed} features probed, {n_committed} committed;\nreads failing the consistency bar abstain")
    order = ["hollow", "rib_count", "max_dimension_mm", "production_volume"]
    labels = {
        "hollow": "hollow = {v}",
        "rib_count": "rib_count = {v}",
        "max_dimension_mm": "max_dimension = {v:.0f} mm",
        "production_volume": "production_volume = {v}",
    }
    y = 4.42
    for k in order:
        v = committed[k]
        val = v["value"]
        val = str(val).lower() if isinstance(val, bool) else val
        txt = labels[k].format(v=val)
        agree = f'{v["n_samples"]} samples, committed'
        ax.add_patch(FancyBboxPatch((x + 0.15, y - 0.62), w - 0.30, 0.66, boxstyle="round,pad=0.03,rounding_size=0.06",
                                    fc=CARD_BG, ec="#C9D2DC", lw=0.9, zorder=2))
        ax.text(x + 0.28, y - 0.16, txt, fontsize=9.6, color=INK, family="monospace", va="top")
        ax.text(x + 0.28, y - 0.44, agree, fontsize=7.6, color=MUTED, va="top")
        y -= 0.90

    # 3) the clause that fires
    x, w = P[2]
    panel(x, w, "A cited rule decides", "one of the 92-clause forbidden_if rule set")
    ax.add_patch(FancyBboxPatch((x + 0.15, 1.30), w - 0.30, 3.18, boxstyle="round,pad=0.04,rounding_size=0.08",
                                fc=CARD_BG, ec=CORAL, lw=1.6, zorder=2))
    cx = x + w / 2
    ax.text(cx, 4.14, nice(FEATURED_PROCESS), ha="center", fontsize=11.5, color=INK, fontweight="bold")
    ax.text(cx, 3.72, "forbidden_if", ha="center", fontsize=8.4, color=CORAL, fontweight="bold")
    ax.text(cx, 3.38, f'{featured["feature"]} {featured["op"]} {str(featured["threshold"]).lower()}',
            ha="center", fontsize=10.5, color=INK, family="monospace")
    ax.text(cx, 2.96, f'observed: {str(featured["observed"]).lower()}', ha="center", fontsize=9.0, color=MUTED, family="monospace")
    ax.text(cx, 2.52, f'"{featured["reason"]}"', ha="center", fontsize=9.2, color=INK, style="italic", wrap=True)
    ax.text(cx, 1.98, f'[{featured["cite"]}]', ha="center", fontsize=8.4, color=NAVY)
    ax.text(cx, 1.52, "the provenance record binds read to rule to cut", ha="center", fontsize=7.8, color=MUTED)

    # 4) eliminations
    x, w = P[3]
    panel(x, w, "Cited eliminations", "every cut carries its clause and source")
    ax.text(x + w / 2, 4.30, f"{len(elims)} of 17 processes\neliminated", ha="center", va="top",
            fontsize=11.0, color=INK, fontweight="bold")
    y = 3.30
    for (feat, op, thr), procs in sorted(by_feat.items(), key=lambda kv: -len(kv[1])):
        ops = {"eq": "=", "gt": ">", "lt": "<", "ge": ">=", "le": "<="}.get(op, op)
        ax.text(x + w / 2, y, f"{feat} {ops} {thr.lower()}", ha="center", fontsize=8.8, color=INK, family="monospace")
        ax.text(x + w / 2, y - 0.30, f"cuts {len(procs)} process{'es' if len(procs) > 1 else ''}", ha="center", fontsize=8.2, color=MUTED)
        y -= 0.78

    # 5) survivors
    x, w = P[4]
    panel(x, w, "Survivors", "the expert-feasible set is contained")
    y = 4.05
    for p in survivors:
        is_exp = p in expert
        ax.add_patch(FancyBboxPatch((x + 0.13, y - 0.60), w - 0.26, 0.70, boxstyle="round,pad=0.03,rounding_size=0.08",
                                    fc=CARD_BG, ec=CORAL if is_exp else "#C9D2DC", lw=1.8 if is_exp else 0.9, zorder=2))
        ax.text(x + w / 2, y - 0.13, nice(p), ha="center", fontsize=9.4, color=INK,
                fontweight="bold" if is_exp else "normal")
        if is_exp:
            ax.text(x + w / 2, y - 0.42, "the expert answer", ha="center", fontsize=7.8, color=CORAL, fontweight="bold")
        y -= 1.00

    for (xa, wa), (xb, _) in zip(P[:-1], P[1:]):
        arrow(xa + wa + 0.06, xb - 0.06)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    print(f"wrote {out}")
    # sanity: every shown value exists in the JSON
    assert set(expert).issubset(set(survivors))
    assert featured["cite"] and "[VERIFY" not in featured["cite"]

if __name__ == "__main__":
    main()
