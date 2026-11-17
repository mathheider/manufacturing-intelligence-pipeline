# -*- coding: utf-8 -*-
"""Does a rule's evidence point the right way on real photos? (dev split only)

For each rule r of process p, its strength s_r(photo) is computed from the
archived perception reads exactly as the soft scorer does. Validity is the
AUC (Mann-Whitney) of s_r for separating:
  favor rules:              photos whose expert set CONTAINS p   (positives)
  disfavor / forbid rules:  photos whose expert set does NOT contain p
AUC 0.5 = the rule's evidence is noise on these photos; > 0.5 = right
direction; < 0.5 = the perceived feature contradicts the rule.

Read the AUC as "right on X% of photo pairs": take one photo where the rule
should fire and one where it should not; how often does the rule give the
stronger evidence to the right one? 50% is a coin flip.

Also a per-feature table: how differently the VLM answers a feature for
photos of processes that should and should not show it.

Writes outputs/rule_validity_v3.json. A rule is "validated" if it fires on
>= MIN_FIRE dev photos and its AUC >= AUC_BAR (both fixed here, before any
test-split scoring).

    python scripts/rule_validity_v3.py [--signatures outputs/signatures_v3.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MIP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MIP / "src"))
from mip import evidence as ev  # noqa: E402

AUC_BAR = 0.55
MIN_FIRE = 5


def auc(pos, neg):
    """P(score_pos > score_neg) + 0.5 P(tie)."""
    if not pos or not neg:
        return None
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    # rank-sum with ties averaged
    ranks, i = {}, 0
    vals = [v for v, _ in allv]
    rsum = 0.0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and vals[j + 1] == vals[i]:
            j += 1
        r = (i + j) / 2 + 1
        rsum += sum(r for k in range(i, j + 1) if allv[k][1] == 1)
        i = j + 1
    n1, n0 = len(pos), len(neg)
    return (rsum - n1 * (n1 + 1) / 2) / (n1 * n0)


def load_evidence(sig_path=None):
    E = json.loads((MIP / "outputs" / "evidence_v2pool.json").read_text(encoding="utf-8"))["photos"]
    if sig_path and Path(sig_path).exists():
        S = json.loads(Path(sig_path).read_text(encoding="utf-8"))["photos"]
        for pid, reads in S.items():
            if pid in E:
                E[pid]["features"].update(reads)
    return E


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", default=str(MIP / "data" / "rules_v3.json"))
    ap.add_argument("--signatures", default=str(MIP / "outputs" / "signatures_v3.json"))
    ap.add_argument("--out", default=str(MIP / "outputs" / "rule_validity_v3.json"))
    ap.add_argument("--tau", type=float, default=0.3)
    a = ap.parse_args(argv)
    E = load_evidence(a.signatures)
    dev = [i for i, v in E.items() if v["split"] == "dev"]
    _, feats, rules = ev.load_rules(a.rules)
    cfg = ev.Config(tau=a.tau)

    rows = []
    for r in rules:
        s = {i: ev.strength(E[i]["features"], r, cfg) for i in dev}
        fires = sum(v > 0 for v in s.values())
        inset = [i for i in dev if r.process in E[i]["expert"]]
        outset = [i for i in dev if r.process not in E[i]["expert"]]
        pos, neg = (inset, outset) if r.kind == "favor" else (outset, inset)
        A = auc([s[i] for i in pos], [s[i] for i in neg]) if fires else None
        rows.append({"id": r.id, "process": r.process, "kind": r.kind,
                     "features": sorted({c.feature for c in r.when}),
                     "observability": r.observability, "fires_on_dev": fires,
                     "auc": None if A is None else round(A, 4),
                     "validated": bool(A is not None and fires >= MIN_FIRE and A >= AUC_BAR)})

    # feature table: best/worst per-process separation of the raw reads
    ftab = {}
    for f in feats:
        def score(i):
            xs = [x for x in E[i]["features"].get(f, []) if x is not None]
            if not xs:
                return None
            if isinstance(xs[0], bool):
                return sum(xs) / len(xs)
            xs = sorted(xs)
            return xs[len(xs) // 2]
        sc = {i: score(i) for i in dev}
        if all(v is None for v in sc.values()):
            ftab[f] = {"has_reads": False}
            continue
        per = {}
        for p in sorted({p for v in E.values() for p in v["expert"]}):
            pos = [sc[i] for i in dev if sc[i] is not None and p in E[i]["expert"]]
            neg = [sc[i] for i in dev if sc[i] is not None and p not in E[i]["expert"]]
            A = auc(pos, neg)
            if A is not None:
                per[p] = round(A, 3)
        best = max(per.items(), key=lambda kv: abs(kv[1] - 0.5))
        ftab[f] = {"has_reads": True, "best_process": best[0], "best_auc": best[1], "per_process_auc": per}

    n_val = sum(r["validated"] for r in rows)
    n_fire = sum(r["fires_on_dev"] > 0 for r in rows)
    summary = {"n_rules": len(rows), "fire_on_dev": n_fire, "validated": n_val,
               "auc_bar": AUC_BAR, "min_fire": MIN_FIRE, "tau": a.tau,
               "validated_by_group": {}, "contradicted (auc<0.45)": sum(1 for r in rows if r["auc"] is not None and r["auc"] < 0.45)}
    for r in rows:
        g = r["id"].split("-")[0]
        summary["validated_by_group"].setdefault(g, [0, 0])
        summary["validated_by_group"][g][1] += 1
        summary["validated_by_group"][g][0] += r["validated"]
    Path(a.out).write_text(json.dumps({"summary": summary, "features": ftab, "rules": rows}, indent=1), encoding="utf-8")
    print(f"{n_fire} of {len(rows)} rules have perception reads on dev photos.")
    print(f"{n_val} point the right way (right on >= {100*AUC_BAR:.0f}% of photo pairs, 50% = coin flip, firing on >= {MIN_FIRE} photos).")
    print(f"{summary['contradicted (auc<0.45)']} point the wrong way (right on < 45% of photo pairs).")
    for g, (v, t) in summary["validated_by_group"].items():
        print(f"  {g}: {v} of {t} point the right way")
    print("How well each feature's answers separate processes (best process; 50% = coin flip):")
    for f, t in ftab.items():
        if t.get("has_reads"):
            print(f"  {f:30s} {t['best_process']:38s} tells apart {100*max(t['best_auc'], 1-t['best_auc']):.0f}% of photo pairs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
