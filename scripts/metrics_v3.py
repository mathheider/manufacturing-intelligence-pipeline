# -*- coding: utf-8 -*-
"""Reframed (additive) metrics over the verified bench-v2 per-instance records.

Reads outputs/records_real_v2.jsonl (cache-only replay, verified to reproduce
benchmark_real_v2.json exactly) and computes the METRICS_MEMO.md framework:
family-level results, confusion/co-occurrence matrices, per-class table,
no-look baselines and the containment-vs-size operating points.

Writes outputs/metrics_v3.json. Never touches preregistered artifacts.
"""
from __future__ import annotations
import json, sys, collections
from pathlib import Path

MIP = Path(__file__).resolve().parent.parent
OUT = MIP / "outputs"

FAM = {
 'Injection_Molding':'Thermoplastic melt','Blow_Molding':'Thermoplastic melt',
 'Thermoforming':'Thermoplastic melt','Overmolding':'Thermoplastic melt',
 'Compression_Molding':'Compression','Sheet_Molding_Compound':'Compression',
 'Bulk_Molding_Compound':'Compression',
 'Resin_Transfer_Molding':'Liquid composite molding',
 'Vacuum_Assisted_RTM':'Liquid composite molding',
 'Structural_Reaction_Injection_Molding':'Liquid composite molding',
 'Hand_Lay_up':'Open-mold composite',
 'Pultrusion':'Continuous profile','Filament_Winding':'Continuous profile',
 'Fused_Filament_Fabrication':'Additive','Stereolithography':'Additive',
 'Selective_Laser_Sintering':'Additive','Laminated_Object_Manufacturing':'Additive'}
FAMILIES = ['Thermoplastic melt','Compression','Liquid composite molding',
            'Open-mold composite','Continuous profile','Additive']
PROCS = sorted(FAM)

def setf1(pred, truth):
    tp=len(pred&truth); fp=len(pred-truth); fn=len(truth-pred)
    p=tp/(tp+fp) if tp+fp else 0.0; r=tp/(tp+fn) if tp+fn else 0.0
    return 2*p*r/(p+r) if p+r else 0.0

def load():
    recs=[]
    for line in (OUT/"records_real_v2.jsonl").read_text(encoding="utf-8").splitlines():
        d=json.loads(line)
        if "_meta" in d: continue
        recs.append(d)
    return recs

def main():
    recs=load()
    by_m=collections.defaultdict(list)
    for r in recs: by_m[r["method"]].append(r)
    methods=sorted(by_m)
    n_all=len(by_m[methods[0]]); n_hard=sum(1 for r in by_m[methods[0]] if r["hard"])
    res={"n_all":n_all,"n_hard":n_hard,"families":FAMILIES,"family_map":FAM}

    # ---- family-level scoring per method
    fam_block={}
    for m,rows in by_m.items():
        for lbl,pool in (("ALL",rows),("HARD",[r for r in rows if r["hard"]])):
            fs=[]; cont=0; sizes=[]; top1fam=0; top1proc=0
            for r in pool:
                ef={FAM[p] for p in r["expert"]}; pf={FAM[p] for p in r["predicted"]}
                fs.append(setf1(pf,ef)); cont+= ef<=pf; sizes.append(len(pf))
                if r["ranked"]:
                    top1fam += FAM[r["ranked"][0]] in ef
                    top1proc += r["ranked"][0] in r["expert"]
            n=len(pool)
            fam_block.setdefault(m,{})[lbl]={
                "family_setF1":round(sum(fs)/n,3),
                "family_containment":round(cont/n,3),
                "mean_family_set_size":round(sum(sizes)/n,2),
                "top1_family_acc":round(top1fam/n,3),
                "top1_process_acc":round(top1proc/n,3)}
    res["family_level"]=fam_block

    # ---- no-look baselines: prevalence top-k containment (process & family)
    prev=collections.Counter()
    for r in by_m[methods[0]]:
        for p in r["expert"]: prev[p]+=1
    order=[p for p,_ in prev.most_common()]
    fam_prev=collections.Counter()
    for r in by_m[methods[0]]:
        for f in {FAM[p] for p in r["expert"]}: fam_prev[f]+=1
    fam_order=[f for f,_ in fam_prev.most_common()]
    def cont_curve(pool, order, mapper=None):
        out={}
        for k in range(1,len(order)+1):
            top=set(order[:k]); c=0
            for r in pool:
                t={mapper[p] for p in r["expert"]} if mapper else set(r["expert"])
                c += t<=top
            out[k]=round(c/len(pool),3)
        return out
    base=by_m[methods[0]]
    res["baseline_prevalence_topk_containment"]={
        "process":{"ALL":cont_curve(base,order),
                   "HARD":cont_curve([r for r in base if r["hard"]],order)},
        "family":{"ALL":cont_curve(base,fam_order,FAM),
                  "HARD":cont_curve([r for r in base if r["hard"]],fam_order,FAM)},
        "process_order":order,"family_order":fam_order}
    # top-1 family accuracy baseline: always most-common family
    for lbl,pool in (("ALL",base),("HARD",[r for r in base if r["hard"]])):
        hit=sum(1 for r in pool if fam_order[0] in {FAM[p] for p in r["expert"]})
        res.setdefault("baseline_top1_family_acc",{})[lbl]=round(hit/len(pool),3)
        hit1=sum(1 for r in pool if order[0] in r["expert"])
        res.setdefault("baseline_top1_process_acc",{})[lbl]=round(hit1/len(pool),3)

    # ---- per-class table (label-based) for each method
    percls={}
    for m,rows in by_m.items():
        tab={}
        for p in PROCS:
            sup=sum(1 for r in rows if p in r["expert"])
            pred=sum(1 for r in rows if p in r["predicted"])
            tp=sum(1 for r in rows if p in r["expert"] and p in r["predicted"])
            rec=tp/sup if sup else 0.0; prec=tp/pred if pred else 0.0
            f1=2*prec*rec/(prec+rec) if prec+rec else 0.0
            tab[p]={"support":sup,"predicted":pred,"precision":round(prec,3),
                    "recall":round(rec,3),"f1":round(f1,3),
                    "false_elim_rate":round(1-rec,3)}
        percls[m]=tab
        macro=sum(v["f1"] for v in tab.values())/len(PROCS)
        percls[m]["_macro_f1_label_based"]=round(macro,3)
    res["per_class"]=percls

    # ---- co-occurrence confusion (rows=expert proc, cols=predicted proc)
    def cooc(rows, mapper=None, labels=None):
        L=labels or PROCS
        M={a:{b:0 for b in L} for a in L}
        rowN={a:0 for a in L}
        for r in rows:
            t={mapper[p] for p in r["expert"]} if mapper else set(r["expert"])
            pr={mapper[p] for p in r["predicted"]} if mapper else set(r["predicted"])
            for a in t:
                rowN[a]+=1
                for b in pr: M[a][b]+=1
        norm={a:{b:(round(M[a][b]/rowN[a],3) if rowN[a] else 0.0) for b in L} for a in L}
        return {"counts":M,"row_norm":norm,"row_support":rowN}
    res["cooccurrence"]={}
    for m in ("Hybrid","Direct","Direct@k","Greedy"):
        res["cooccurrence"][m]={
            "process":cooc(by_m[m]),
            "family":cooc(by_m[m],FAM,FAMILIES)}

    # ---- family top-1 confusion on single-family instances (clean rows)
    def fam_top1_cm(rows):
        M={a:{b:0 for b in FAMILIES} for a in FAMILIES}
        used=0
        for r in rows:
            ef={FAM[p] for p in r["expert"]}
            if len(ef)!=1 or not r["ranked"]: continue
            used+=1
            M[next(iter(ef))][FAM[r["ranked"][0]]]+=1
        return {"counts":M,"n_single_family":used}
    res["family_top1_cm"]={m:fam_top1_cm(by_m[m]) for m in ("Hybrid","Direct")}

    # ---- error locality: when top-1 process is wrong, is its family right?
    loc={}
    for m,rows in by_m.items():
        for lbl,pool in (("ALL",rows),("HARD",[r for r in rows if r["hard"]])):
            wrong=[r for r in pool if r["ranked"] and r["ranked"][0] not in r["expert"]]
            if not wrong: continue
            samefam=sum(1 for r in wrong
                        if FAM[r["ranked"][0]] in {FAM[p] for p in r["expert"]})
            loc.setdefault(m,{})[lbl]={"n_top1_errors":len(wrong),
                "sibling_share":round(samefam/len(wrong),3)}
    res["error_locality"]=loc

    # ---- per-family method results (case-study grid)
    perfam={}
    for m,rows in by_m.items():
        for famname in FAMILIES:
            pool=[r for r in rows if famname in {FAM[p] for p in r["expert"]}]
            if not pool: continue
            n=len(pool)
            cont=sum(1 for r in pool if set(r["expert"])<=set(r["predicted"]))/n
            fcont=sum(1 for r in pool
                      if {FAM[p] for p in r["expert"]}<={FAM[p] for p in r["predicted"]})/n
            f1=sum(r["f1"] for r in pool)/n
            top1=sum(1 for r in pool if r["ranked"] and FAM[r["ranked"][0]] in {FAM[p] for p in r["expert"]})/n
            perfam.setdefault(m,{})[famname]={"n":n,"setF1":round(f1,3),
                "containment":round(cont,3),"family_containment":round(fcont,3),
                "top1_family_acc":round(top1,3)}
    res["per_family"]=perfam

    (OUT/"metrics_v3.json").write_text(json.dumps(res,indent=2),encoding="utf-8")
    print("wrote outputs/metrics_v3.json")
    print(json.dumps({"family_level":fam_block,
                      "baseline_top1_family_acc":res["baseline_top1_family_acc"],
                      "error_locality":loc},indent=1)[:3000])

if __name__=="__main__":
    sys.exit(main())
