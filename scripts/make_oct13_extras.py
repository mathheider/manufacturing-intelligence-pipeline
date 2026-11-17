# -*- coding: utf-8 -*-
"""Extras for the Oct 13, 2026 update deck (Update_Oct13_2026.pptx, folder root).

Reads outputs/metrics_v3.json + outputs/records_real_v2.jsonl (the Sep 1 cache-only replay of
bench-v2; nothing here calls a model). Writes two new artifacts and rewrites nothing:

  outputs/family_case_studies_v3.json
      Slide 8. Top-1 family accuracy per process family for all four variants, 95% bootstrap CIs
      (4,000 resamples of parts within each family, seed 20261013), and Spearman rank agreement
      across the six families. Accuracies are asserted equal to metrics_v3.json["per_family"].
  outputs/assets/metrics_v3/fig6_process_cooccurrence.png
      Backup slide 14. 17x17 row-normalized co-occurrence, Direct vs Hybrid, from
      metrics_v3.json["cooccurrence"][m]["process"]; three cells spot-checked against the records.
"""
import json, random, collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MIP = Path(__file__).resolve().parent.parent
OUT = MIP/"outputs"; ASSETS = OUT/"assets"/"metrics_v3"; ASSETS.mkdir(parents=True, exist_ok=True)
MV = json.loads((OUT/"metrics_v3.json").read_text(encoding="utf-8"))
recs = [json.loads(l) for l in (OUT/"records_real_v2.jsonl").read_text(encoding="utf-8").splitlines()]
recs = [r for r in recs if "_meta" not in r]
FAMS = MV["families"]; fmap = MV["family_map"]
METHODS = ["Direct", "Direct@k", "Greedy", "Hybrid"]
by = collections.defaultdict(dict)
for r in recs: by[r["method"]][r["id"]] = r
ids = list(by["Direct"])

# ---------------- slide 8: per-family case studies -------------------------
SEED, B = 20261013, 4000
random.seed(SEED)
table = {}
for m in METHODS:
    for F in FAMS:
        sub = [by[m][i] for i in ids if F in {fmap[q] for q in by[m][i]["expert"]}]
        hits = [1.0 if r["ranked"] and fmap[r["ranked"][0]] in {fmap[q] for q in r["expert"]} else 0.0 for r in sub]
        acc = sum(hits)/len(hits)
        assert abs(acc - MV["per_family"][m][F]["top1_family_acc"]) < 6e-4, (m, F)
        assert len(sub) == MV["per_family"][m][F]["n"], (m, F)
        bs = sorted(sum(random.choice(hits) for _ in hits)/len(hits) for _ in range(B))
        table.setdefault(F, {"n": len(sub)})[m] = {"top1_family_acc": round(acc, 6),
                                                   "ci95": [round(bs[int(0.025*B)], 6), round(bs[int(0.975*B)-1], 6)]}

def ranks(v):
    o = sorted(range(len(v)), key=lambda i: v[i]); rk = [0.0]*len(v); i = 0
    while i < len(o):
        j = i
        while j+1 < len(o) and v[o[j+1]] == v[o[i]]: j += 1
        for k in range(i, j+1): rk[o[k]] = (i+j)/2 + 1
        i = j+1
    return rk
def spearman(a, b):
    ra, rb = ranks(a), ranks(b); n = len(a); ma, mb = sum(ra)/n, sum(rb)/n
    num = sum((x-ma)*(y-mb) for x, y in zip(ra, rb))
    return num/(sum((x-ma)**2 for x in ra)*sum((y-mb)**2 for y in rb))**0.5
vec = {m: [table[F][m]["top1_family_acc"] for F in FAMS] for m in METHODS}
rho = {a: {b: round(spearman(vec[a], vec[b]), 4) for b in METHODS} for a in METHODS}
(OUT/"family_case_studies_v3.json").write_text(json.dumps({
    "_meta": {"source": "outputs/records_real_v2.jsonl (Sep 1 cache-only replay); accuracies asserted equal to metrics_v3.json per_family",
              "metric": "top-1 family accuracy: rank-1 process's family is among the expert set's families",
              "rows_overlap": "a part whose expert set spans two families counts in both rows",
              "bootstrap": {"resamples": B, "seed": SEED, "unit": "parts within each family"},
              "spearman": "rank agreement across the six families (descriptive; n=6)",
              "deck": "Update_Oct13_2026.pptx slide 8"},
    "families": table, "spearman_across_families": rho}, indent=2), encoding="utf-8")

# ---------------- backup slide 14: 17x17 co-occurrence --------------------
INK="#0b0b0b"; INK2="#52514e"; MUT="#9a9891"; SURF="#fcfcfb"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":12,"axes.edgecolor":MUT,"axes.labelcolor":INK2,
    "xtick.color":INK2,"ytick.color":INK2,"figure.facecolor":SURF,"axes.facecolor":SURF,
    "axes.spines.top":False,"axes.spines.right":False})
sup = MV["cooccurrence"]["Direct"]["process"]["row_support"]
order = []
for F in FAMS:
    order += sorted([q for q in fmap if fmap[q] == F], key=lambda q: -sup[q])
SHORT = {"Injection_Molding":"Injection molding","Blow_Molding":"Blow molding","Thermoforming":"Thermoforming",
 "Overmolding":"Overmolding","Compression_Molding":"Compression molding","Sheet_Molding_Compound":"SMC",
 "Bulk_Molding_Compound":"BMC","Resin_Transfer_Molding":"RTM","Vacuum_Assisted_RTM":"VARTM",
 "Structural_Reaction_Injection_Molding":"SRIM","Hand_Lay_up":"Hand lay-up","Pultrusion":"Pultrusion",
 "Filament_Winding":"Filament winding","Fused_Filament_Fabrication":"FFF","Stereolithography":"SLA",
 "Selective_Laser_Sintering":"SLS","Laminated_Object_Manufacturing":"LOM"}
def cell(m, a, b):
    rows = [r for r in by[m].values() if a in r["expert"]]
    return sum(b in r["predicted"] for r in rows)/len(rows)
for m, a, b in [("Direct","Blow_Molding","Injection_Molding"),("Hybrid","Hand_Lay_up","Fused_Filament_Fabrication"),("Direct","Pultrusion","Pultrusion")]:
    assert abs(cell(m, a, b) - MV["cooccurrence"][m]["process"]["row_norm"][a][b]) < 6e-4, (m, a, b)
bounds = []; k = 0
for F in FAMS:
    k += sum(1 for q in order if fmap[q] == F); bounds.append(k)
def heat(ax, m, title):
    N = MV["cooccurrence"][m]["process"]["row_norm"]
    M = np.array([[N[a][b] for b in order] for a in order])
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(17)); ax.set_xticklabels([SHORT[q] for q in order], rotation=55, ha="right", fontsize=9.5)
    ax.set_yticks(range(17)); ax.set_yticklabels([f"{SHORT[q]} (n={sup[q]})" for q in order], fontsize=9.5)
    for kk in bounds[:-1]:
        ax.axhline(kk-0.5, color=INK2, lw=1.1); ax.axvline(kk-0.5, color=INK2, lw=1.1)
    for i_, a in enumerate(order):
        for j_, b in enumerate(order):
            v = M[i_, j_]
            if v >= 0.15:
                ax.text(j_, i_, f"{v:.2f}", ha="center", va="center", fontsize=6.9, color=("white" if v > 0.55 else INK))
    ax.set_title(title, loc="left", color=INK, fontsize=12.5, pad=8)
    return im
fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.6))
fig.subplots_adjust(wspace=0.55, left=0.145, right=0.935, bottom=0.215, top=0.90)
heat(axes[0], "Direct", "Direct: the Injection-molding column absorbs most rows")
im = heat(axes[1], "Hybrid", "Hybrid: FFF survives for nearly every row")
cb = fig.colorbar(im, ax=axes, fraction=0.022, pad=0.015)
cb.set_label("P(process among predictions | process in expert set)", fontsize=9.5); cb.outline.set_visible(False)
fig.savefig(ASSETS/"fig6_process_cooccurrence.png", dpi=200, facecolor=SURF, bbox_inches="tight", pad_inches=0.15)
plt.close(fig)
print("wrote", OUT/"family_case_studies_v3.json", "and", ASSETS/"fig6_process_cooccurrence.png")
print("spearman vs Direct:", {m: rho["Direct"][m] for m in METHODS}, "| Greedy vs Hybrid:", rho["Greedy"]["Hybrid"])
