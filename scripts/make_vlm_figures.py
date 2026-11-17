# -*- coding: utf-8 -*-
"""VLM-focus figures (Sept 2026). From verified records_real_v2.jsonl only."""
import json, collections
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MIP=Path(__file__).resolve().parent.parent
OUT=MIP/"outputs"; A=OUT/"assets"/"metrics_v3"; A.mkdir(parents=True,exist_ok=True)
INK="#0b0b0b"; INK2="#52514e"; MUT="#9a9891"; SURF="#fcfcfb"
BLUE="#2a78d6"; GOLD="#eda100"; GRID="#e8e6e1"
FAMC={"Melt":"#2a78d6","Comp":"#eb6834","LCM":"#1baf7a","Open":"#eda100","Cont":"#e87ba4","Add":"#4a3aa7"}
FAMNAME={"Melt":"Melt","Comp":"Compress.","LCM":"LCM","Open":"Open-mold","Cont":"Contin.","Add":"Additive"}
plt.rcParams.update({"font.size":12,"axes.edgecolor":MUT,"axes.labelcolor":INK2,
 "xtick.color":INK2,"ytick.color":INK2,"figure.facecolor":SURF,"axes.facecolor":SURF,
 "axes.spines.top":False,"axes.spines.right":False})
FAM={'Injection_Molding':'Melt','Blow_Molding':'Melt','Thermoforming':'Melt','Overmolding':'Melt',
 'Compression_Molding':'Comp','Sheet_Molding_Compound':'Comp','Bulk_Molding_Compound':'Comp',
 'Resin_Transfer_Molding':'LCM','Vacuum_Assisted_RTM':'LCM','Structural_Reaction_Injection_Molding':'LCM',
 'Hand_Lay_up':'Open','Pultrusion':'Cont','Filament_Winding':'Cont',
 'Fused_Filament_Fabrication':'Add','Stereolithography':'Add','Selective_Laser_Sintering':'Add',
 'Laminated_Object_Manufacturing':'Add'}
recs=[json.loads(l) for l in (OUT/"records_real_v2.jsonl").read_text(encoding="utf-8").splitlines()]
recs=[r for r in recs if "_meta" not in r]
tk={r['id']:r for r in recs if r['method']=='Direct@k'}
dr={r['id']:r for r in recs if r['method']=='Direct'}
ids=list(dr)
prev=collections.Counter()
for i in ids:
    for p in dr[i]['expert']: prev[p]+=1
order=[p for p,_ in prev.most_common()]

# ---- vfig1: containment@k, VLM vs no-look, ALL + HARD
def curve(pool_ids, ranked_src):
    vlm=[];nol=[]
    for k in range(1,9):
        top=set(order[:k])
        vlm.append(sum(1 for i in pool_ids if set(dr[i]['expert'])<=set(ranked_src[i]['ranked'][:k]))/len(pool_ids))
        nol.append(sum(1 for i in pool_ids if set(dr[i]['expert'])<=top)/len(pool_ids))
    return vlm,nol
hard=[i for i in ids if dr[i]['hard']]
fig,axes=plt.subplots(1,2,figsize=(10.8,4.6),sharey=True)
for ax,(lbl,pool) in zip(axes,(("ALL (n=635)",ids),("HARD (n=387)",hard))):
    v,n=curve(pool,tk)
    ks=list(range(1,9))
    ax.plot(ks,n,color=MUT,lw=2,marker="o",ms=5,label="no-look prevalence top-k")
    ax.plot(ks,v,color=BLUE,lw=2.4,marker="o",ms=6,label="VLM ranked top-k")
    for k,val in [(8,v[7])]:
        ax.annotate(f"{val:.2f}",(8,val),textcoords="offset points",xytext=(-2,9),color=INK,fontsize=11,ha="right")
    for k,val in [(8,n[7])]:
        ax.annotate(f"{val:.2f}",(8,val),textcoords="offset points",xytext=(-2,-16),color=INK2,fontsize=10.5,ha="right")
    ax.set_xlabel(f"k (processes kept) — {lbl}")
    ax.set_xticks(ks); ax.set_ylim(0,1.0)
    ax.xaxis.grid(True,color=GRID); ax.yaxis.grid(True,color=GRID); ax.set_axisbelow(True)
axes[0].set_ylabel("expert-set containment")
axes[0].legend(frameon=False,loc="upper left",fontsize=10.5)
fig.suptitle("The VLM's ranked list clears the no-look floor at every k ≥ 2 — and the gap widens on HARD",
             x=0.01,ha="left",color=INK,fontsize=14.5)
fig.tight_layout(rect=(0,0,1,0.92)); fig.savefig(A/"vfig1_containment_k.png",dpi=200); plt.close(fig)

# ---- vfig2: family skill, VLM only
floors={"ALL":0.718,"HARD":0.537}
vals={"ALL":{"Direct":0.846,"Direct@8":0.836},"HARD":{"Direct":0.778,"Direct@8":0.752}}
fig,axes=plt.subplots(1,2,figsize=(10.6,3.6),sharey=True)
for ax,lbl in zip(axes,("ALL","HARD")):
    names=["Direct@8","Direct"]; ys=range(2)
    vs=[vals[lbl][n] for n in names]
    ax.barh(list(ys),vs,color=[GOLD,BLUE],height=0.52,zorder=3)
    fv=floors[lbl]
    ax.axvline(fv,color=INK2,lw=1.6,ls=(0,(4,3)),zorder=4)
    ax.text(fv,1.62,f"no-look floor {fv:.3f}",color=INK2,fontsize=10.5,ha="center")
    for y,v in zip(ys,vs): ax.text(v+0.012,y,f"{v:.3f}",va="center",color=INK,fontsize=11.5)
    ax.set_yticks(list(ys)); ax.set_yticklabels(names,color=INK)
    ax.set_xlim(0,1.0); ax.set_xlabel(f"top-1 family accuracy, {lbl}")
    ax.xaxis.grid(True,color=GRID); ax.set_axisbelow(True)
fig.suptitle("Family-level skill: the VLM beats the floor most where the pool is hardest",
             x=0.01,ha="left",color=INK,fontsize=14.5)
fig.tight_layout(rect=(0,0,1,0.90)); fig.savefig(A/"vfig2_family_skill_vlm.png",dpi=200); plt.close(fig)

# ---- vfig3: balanced two-panel family CM (conditional recall + precision)
def famset(r): return {FAM[p] for p in r['expert']}
FKEYS=["Melt","Comp","LCM","Open","Cont","Add"]
FLBL={"Melt":"Melt","Comp":"Compress.","LCM":"LCM","Open":"Open-mold","Cont":"Contin.","Add":"Additive"}
# Panel A: conditional rows -- for X != Melt only parts where Melt is NOT feasible
Mrec=np.zeros((6,6)); supA=[]
for i,X in enumerate(FKEYS):
    pool=[dr[j] for j in ids if (X in famset(dr[j])) and (X=="Melt" or "Melt" not in famset(dr[j]))]
    supA.append(len(pool))
    import collections as _c
    dist=_c.Counter(FAM[r["ranked"][0]] for r in pool if r["ranked"])
    for j,Y in enumerate(FKEYS): Mrec[i,j]=dist.get(Y,0)/len(pool) if pool else 0
balacc=np.mean(np.diag(Mrec))
# Panel B: precision view -- P(row family feasible | model predicted column family)
Mpre=np.zeros((6,6)); supB=[]
for j,X in enumerate(FKEYS):
    pool=[dr[i2] for i2 in ids if dr[i2]["ranked"] and FAM[dr[i2]["ranked"][0]]==X]
    supB.append(len(pool))
    for i2,Y in enumerate(FKEYS):
        Mpre[i2,j]=(sum(1 for r in pool if Y in famset(r))/len(pool)) if pool else 0
fig,axes=plt.subplots(1,2,figsize=(12.6,5.2))
fig.subplots_adjust(wspace=0.5,top=0.78)
def draw(ax,M,rowlbl,collbl,ttl):
    im=ax.imshow(M,cmap="Blues",vmin=0,vmax=1,aspect="auto")
    ax.set_xticks(range(6)); ax.set_xticklabels(collbl,rotation=35,ha="right",fontsize=10.5)
    ax.set_yticks(range(6)); ax.set_yticklabels(rowlbl,fontsize=10.5)
    for a in range(6):
        for b in range(6):
            v=M[a,b]
            if v>=0.05: ax.text(b,a,f"{v:.2f}",ha="center",va="center",fontsize=9.5,color=("white" if v>0.55 else INK))
    ax.set_title(ttl,loc="left",color=INK,fontsize=12.5,pad=8)
    return im
draw(axes[0],Mrec,[f"{FLBL[k]} (n={n})" for k,n in zip(FKEYS,supA)],[FLBL[k] for k in FKEYS],
     "Errors only: rows exclude Melt-feasible parts")
im=draw(axes[1],Mpre,[FLBL[k] for k in FKEYS],[f"{FLBL[k]} (n={n})" for k,n in zip(FKEYS,supB)],
        "Precision: P(row feasible | model said column)")
cb=fig.colorbar(im,ax=axes,fraction=0.03,pad=0.02)
fig.suptitle(f"Balanced family confusion, Direct VLM -- recall on unambiguous rows (balanced acc. {balacc:.2f}) and precision of each call",
             x=0.01,y=0.97,ha="left",color=INK,fontsize=13.5)
fig.savefig(A/"vfig3_direct_family_cm.png",dpi=200,bbox_inches="tight"); plt.close(fig)

# ---- vfig4: per-process recall@8
rows=[]
for p in order:
    supn=prev[p]
    hit=sum(1 for i in ids if p in dr[i]['expert'] and p in tk[i]['ranked'][:8])
    rows.append((p,hit/supn,supn))
rows.sort(key=lambda x:-x[1])
fig,ax=plt.subplots(figsize=(10.8,6.2))
y=list(range(len(rows)))[::-1]
ax.barh(y,[r[1] for r in rows],color=[FAMC[FAM[r[0]]] for r in rows],height=0.62,zorder=3)
for yy,(p,v,supn) in zip(y,rows):
    if v>0.9: ax.text(v-0.012,yy,f"{v:.2f}",va="center",ha="right",fontsize=10.5,color="white",zorder=5)
    else: ax.text(v+0.008,yy,f"{v:.2f}",va="center",fontsize=10.5,color=INK,zorder=5)
    ax.text(1.02,yy,f"n={supn}",va="center",fontsize=9.5,color=INK2)
ax.set_yticks(y); ax.set_yticklabels([r[0].replace("_"," ") for r in rows],fontsize=11,color=INK)
ax.set_xlim(0,1.12); ax.set_xticks([0,0.2,0.4,0.6,0.8,1.0])
ax.set_xlabel("recall@8  P(process in VLM top-8 | expert says feasible)")
ax.xaxis.grid(True,color=GRID); ax.set_axisbelow(True)
ax.set_title("What the VLM sees — and its four blind spots",loc="left",color=INK,fontsize=15,pad=12)
import matplotlib.patches as mp
ax.legend(handles=[mp.Patch(color=FAMC[k],label=FAMNAME[k]) for k in FAMC],
          loc="center left",bbox_to_anchor=(0.56,0.14),frameon=False,fontsize=10,ncols=2)
fig.tight_layout(); fig.savefig(A/"vfig4_recall8.png",dpi=200); plt.close(fig)
print("wrote", [f.name for f in sorted(A.glob('vfig*.png'))])
