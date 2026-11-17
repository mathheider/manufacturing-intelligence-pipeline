# -*- coding: utf-8 -*-
"""Figures for the Sept 2026 metrics-rework update deck.

Reads outputs/metrics_v3.json + outputs/records_real_v2.jsonl (both derived
from the verified cache-only replay). Writes PNGs to outputs/assets/metrics_v3/.
"""
import json, collections
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MIP = Path(__file__).resolve().parent.parent
OUT = MIP/"outputs"; ASSETS = OUT/"assets"/"metrics_v3"; ASSETS.mkdir(parents=True, exist_ok=True)
MV = json.loads((OUT/"metrics_v3.json").read_text(encoding="utf-8"))

INK="#0b0b0b"; INK2="#52514e"; MUT="#9a9891"; SURF="#fcfcfb"
C_DIRECT="#2a78d6"; C_GREEDY="#1baf7a"; C_HYBRID="#eb6834"; C_TOPK="#eda100"; C_BASE="#b7b5ad"
FAMC={"Thermoplastic melt":"#2a78d6","Compression":"#eb6834","Liquid composite molding":"#1baf7a",
      "Open-mold composite":"#eda100","Continuous profile":"#e87ba4","Additive":"#4a3aa7"}
plt.rcParams.update({"font.size":12,"axes.edgecolor":MUT,"axes.labelcolor":INK2,
    "xtick.color":INK2,"ytick.color":INK2,"figure.facecolor":SURF,"axes.facecolor":SURF,
    "axes.spines.top":False,"axes.spines.right":False})

recs=[json.loads(l) for l in (OUT/"records_real_v2.jsonl").read_text(encoding="utf-8").splitlines()]
recs=[r for r in recs if "_meta" not in r]
by_m=collections.defaultdict(list)
for r in recs: by_m[r["method"]].append(r)

def setf1(pred,truth):
    tp=len(pred&truth);fp=len(pred-truth);fn=len(truth-pred)
    p=tp/(tp+fp) if tp+fp else 0;q=tp/(tp+fn) if tp+fn else 0
    return 2*p*q/(p+q) if p+q else 0

# frozen headline F1s from benchmark_real_v2.json
BR=json.loads((OUT/"benchmark_real_v2.json").read_text(encoding="utf-8"))
def f1of(m,blk): return BR[blk][m]["f1"]

# no-look baselines recomputed from expert sets
base_pool=by_m["Direct"]
def base_f1(predset,pool): return sum(setf1(predset,set(r["expert"])) for r in pool)/len(pool)
hardpool=[r for r in base_pool if r["hard"]]
order=MV["baseline_prevalence_topk_containment"]["process_order"]
CONST={"ALL":base_f1({"Injection_Molding"},base_pool),"HARD":base_f1({"Injection_Molding"},hardpool)}
TOP3={"ALL":base_f1(set(order[:3]),base_pool),"HARD":base_f1(set(order[:3]),hardpool)}

# ---------------- fig 1: the metric has no floor (set-F1, HARD) -------------
rows=[("Always 'Injection Molding'\n(looks at nothing)",CONST["HARD"],C_BASE),
      ("Top-3 by prevalence\n(looks at nothing)",TOP3["HARD"],C_BASE),
      ("Direct VLM",f1of("Direct","hard_subset"),C_DIRECT),
      ("Direct@8",f1of("Direct@k","hard_subset"),C_TOPK),
      ("Greedy",f1of("Greedy","hard_subset"),C_GREEDY),
      ("Hybrid",f1of("Hybrid","hard_subset"),C_HYBRID)]
rows=rows[::-1]
fig,ax=plt.subplots(figsize=(9.6,4.6))
y=range(len(rows))
ax.barh(y,[r[1] for r in rows],color=[r[2] for r in rows],height=0.62,zorder=3)
for i,(lab,v,c) in enumerate(rows):
    ax.text(v+0.008,i,f"{v:.3f}",va="center",color=INK,fontsize=12,zorder=4)
ax.set_yticks(list(y)); ax.set_yticklabels([r[0] for r in rows],color=INK)
ax.set_xlim(0,0.62); ax.set_xlabel("set-F1, HARD subset (n=387)")
ax.xaxis.grid(True,color="#e8e6e1",zorder=0); ax.set_axisbelow(True)
ax.set_title("The current headline metric has no floor",loc="left",color=INK,fontsize=15,pad=14)
ax.text(0,1.02,"zero-knowledge predictors score in the same range as the pipelines",
        transform=ax.transAxes,color=INK2,fontsize=11.5)
fig.tight_layout(); fig.savefig(ASSETS/"fig1_no_floor.png",dpi=200); plt.close(fig)

# ---------------- fig 2: family-level skill (top-1 family accuracy) ---------
fl=MV["family_level"]; floor=MV["baseline_top1_family_acc"]
meths=[("Direct",C_DIRECT),("Direct@k",C_TOPK),("Greedy",C_GREEDY),("Hybrid",C_HYBRID)]
fig,axes=plt.subplots(1,2,figsize=(10.6,4.4),sharey=True)
for ax,lbl in zip(axes,("ALL","HARD")):
    names=[m for m,_ in meths][::-1]
    vals=[fl[m][lbl]["top1_family_acc"] for m,_ in meths][::-1]
    cols=[c for _,c in meths][::-1]
    ax.barh(range(4),vals,color=cols,height=0.58,zorder=3)
    fv=floor[lbl]
    ax.axvline(fv,color=INK2,lw=1.6,ls=(0,(4,3)),zorder=4)
    ax.text(fv,3.72,f"no-look floor {fv:.3f}",color=INK2,fontsize=10.5,ha="center")
    for i,v in enumerate(vals):
        ax.text(v+0.012,i,f"{v:.3f}",va="center",color=INK,fontsize=11.5,zorder=4)
    ax.set_yticks(range(4)); ax.set_yticklabels(["Direct@8" if n=="Direct@k" else n for n in names],color=INK)
    ax.set_xlim(0,1.0); ax.set_xlabel(f"top-1 family accuracy, {lbl}" + (" (n=635)" if lbl=="ALL" else " (n=387)"))
    ax.xaxis.grid(True,color="#e8e6e1",zorder=0); ax.set_axisbelow(True)
fig.suptitle("Measured at the family level, the VLM clearly beats the floor -- the engine does not",
             x=0.01,ha="left",color=INK,fontsize=15)
fig.tight_layout(rect=(0,0,1,0.93)); fig.savefig(ASSETS/"fig2_family_skill.png",dpi=200); plt.close(fig)

# ---------------- fig 3: family co-occurrence heatmaps ----------------------
FAMS=MV["families"]
short={"Thermoplastic melt":"Melt","Compression":"Compress.","Liquid composite molding":"LCM",
       "Open-mold composite":"Open-mold","Continuous profile":"Contin.","Additive":"Additive"}
def heat(ax,m,title):
    cm=MV["cooccurrence"][m]["family"]; N=cm["row_norm"]; sup=cm["row_support"]
    import numpy as np
    M=np.array([[N[a][b] for b in FAMS] for a in FAMS])
    im=ax.imshow(M,cmap="Blues",vmin=0,vmax=1,aspect="auto")
    ax.set_xticks(range(6)); ax.set_xticklabels([short[f] for f in FAMS],rotation=35,ha="right",fontsize=10.5)
    ax.set_yticks(range(6)); ax.set_yticklabels([f"{short[a]} (n={sup[a]})" for a in FAMS],fontsize=10.5)
    for i,a in enumerate(FAMS):
        for j,b in enumerate(FAMS):
            v=M[i,j]
            if v>=0.05:
                ax.text(j,i,f"{v:.2f}",ha="center",va="center",fontsize=9.5,
                        color=("white" if v>0.55 else INK))
    ax.set_title(title,loc="left",color=INK,fontsize=13,pad=8)
    return im
fig,axes=plt.subplots(1,2,figsize=(12.4,5.1))
fig.subplots_adjust(wspace=0.55,top=0.80)
heat(axes[0],"Direct","Direct: answers pool in Melt -- the prior, made visible")
im=heat(axes[1],"Hybrid","Hybrid: everything drains toward Additive survivors")
cb=fig.colorbar(im,ax=axes,fraction=0.03,pad=0.02); cb.set_label("P(family among predictions | family in expert set)",fontsize=9.5)
fig.suptitle("The clustering structure, family level  (row = expert family membership, col = predicted / surviving family)",
             x=0.01,y=0.97,ha="left",color=INK,fontsize=14)
fig.savefig(ASSETS/"fig3_family_confusion.png",dpi=200,bbox_inches="tight"); plt.close(fig)

# ---------------- fig 4: Hybrid false-elimination by process ----------------
pc=MV["per_class"]["Hybrid"]; fammap=MV["family_map"]
items=[(p,v) for p,v in pc.items() if not p.startswith("_")]
items.sort(key=lambda x:-x[1]["false_elim_rate"])
CL={"Injection_Molding":6,"Thermoforming":6,"Compression_Molding":7,"Resin_Transfer_Molding":5,
"Vacuum_Assisted_RTM":4,"Structural_Reaction_Injection_Molding":6,"Sheet_Molding_Compound":7,
"Bulk_Molding_Compound":6,"Hand_Lay_up":5,"Blow_Molding":7,"Pultrusion":5,"Filament_Winding":5,
"Overmolding":5,"Fused_Filament_Fabrication":4,"Stereolithography":4,"Selective_Laser_Sintering":4,
"Laminated_Object_Manufacturing":6}
labels=[p.replace("_"," ")+f"  ({CL[p]} cl.)" for p,_ in items]
vals=[v["false_elim_rate"] for _,v in items]
cols=[FAMC[fammap[p]] for p,_ in items]
sups=[v["support"] for _,v in items]
fig,ax=plt.subplots(figsize=(10.8,6.4))
y=range(len(items))[::-1]
ax.barh(list(y),vals,color=cols,height=0.62,zorder=3)
for yy,v,sp in zip(y,vals,sups):
    if v>0.9:
        ax.text(v-0.012,yy,f"{v:.2f}",va="center",ha="right",fontsize=10.5,color="white",zorder=5)
    else:
        ax.text(v+0.008,yy,f"{v:.2f}",va="center",fontsize=10.5,color=INK,zorder=5)
    ax.text(1.02,yy,f"n={sp}",va="center",fontsize=9.5,color=INK2)
ax.set_yticks(list(y)); ax.set_yticklabels(labels,fontsize=11,color=INK)
ax.set_xlim(0,1.12); ax.set_xticks([0,0.2,0.4,0.6,0.8,1.0])
ax.set_xlabel("false-elimination rate  P(process eliminated | expert says feasible)")
ax.xaxis.grid(True,color="#e8e6e1",zorder=0); ax.set_axisbelow(True)
ax.set_title("Where Hybrid actually fails: which feasible processes get eliminated",
             loc="left",color=INK,fontsize=15,pad=12)
import matplotlib.patches as mpatches
hand=[mpatches.Patch(color=FAMC[f],label=short[f]+f"  ({'4' if f=='Additive' else '4-7'} clauses)") for f in FAMS]
hand=[mpatches.Patch(color=FAMC[f],label=short[f]) for f in FAMS]
ax.legend(handles=hand,loc="center left",bbox_to_anchor=(0.40,0.15),frameon=False,fontsize=10,ncols=2)
fig.tight_layout(); fig.savefig(ASSETS/"fig4_false_elim.png",dpi=200); plt.close(fig)

# ---------------- fig 5: operating curve (containment vs set size, HARD) ----
curve=MV["baseline_prevalence_topk_containment"]["process"]["HARD"]
ks=sorted(int(k) for k in curve); xs=ks; ys=[curve[str(k)] for k in ks]
pts=[("Direct",1.0,BR["hard_subset"]["Direct"]["containment"],C_DIRECT),
     ("Greedy",BR["hard_subset"]["Greedy"]["avg_set_size"],BR["hard_subset"]["Greedy"]["containment"],C_GREEDY),
     ("Hybrid",BR["hard_subset"]["Hybrid"]["avg_set_size"],BR["hard_subset"]["Hybrid"]["containment"],C_HYBRID),
     ("Direct@8",8.0,BR["hard_subset"]["Direct@k"]["containment"],C_TOPK)]
fig,ax=plt.subplots(figsize=(9.8,5.2))
ax.plot(xs,ys,color=MUT,lw=2,zorder=2,label="no-look prevalence top-k")
ax.fill_between(xs,0,ys,color="#ecebe7",zorder=1)
for name,x,yv,c in pts:
    ax.scatter([x],[yv],s=110,color=c,zorder=4,edgecolor=SURF,linewidth=2)
    dy=0.045 if name!="Greedy" else -0.07
    ax.annotate(f"{name}  ({yv:.2f} @ {x:.1f})",(x,yv),textcoords="offset points",
                xytext=(10,8 if name!="Greedy" else -16),fontsize=11.5,color=INK)
ax.text(10.5,0.30,"shaded = achievable while\nignoring the image",color=INK2,fontsize=10.5)
ax.set_xlabel("mean survivor-set size (processes kept, of 17)")
ax.set_ylabel("expert-set containment (no false elimination)")
ax.set_xlim(0.5,17.5); ax.set_ylim(0,1.0)
ax.xaxis.grid(True,color="#e8e6e1"); ax.yaxis.grid(True,color="#e8e6e1"); ax.set_axisbelow(True)
ax.set_title("Safe narrowing, HARD subset: above the gray curve = adds knowledge",
             loc="left",color=INK,fontsize=14.5,pad=12)
ax.legend(frameon=False,loc="lower right",fontsize=10.5)
fig.tight_layout(); fig.savefig(ASSETS/"fig5_operating_curve.png",dpi=200); plt.close(fig)

print("floor values:",{"const":CONST,"top3":TOP3})
for f in sorted(ASSETS.iterdir()): print(f.name)
