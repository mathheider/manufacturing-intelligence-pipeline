# -*- coding: utf-8 -*-
"""Exploratory, DEV ONLY (535 photos), zero API calls. Gold-100 is not read.

E1  trust signal: does rule evidence tell us when the VLM's top pick is right?
E2  learned rule weights: 5-fold cross-validated logistic model on (photo, process) pairs
      M0 VLM ranking | M1 VLM signals + process base rates (no rules) | M2 = M1 + rules
"""
import json, sys, random
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

MIP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MIP / "src")); sys.path.insert(0, str(MIP / "scripts"))
from mip import evidence as ev
import eval_soft_v3 as es

E, fam, _ = es.load(None)
dev = sorted(i for i, v in E.items() if v["split"] == "dev")       # test photos never touched
procs = sorted({p for v in E.values() for p in v["top8"]} | set(fam))
P = len(procs); pidx = {p: k for k, p in enumerate(procs)}
_, _, rules = ev.load_rules(MIP / "data" / "rules_v3.json")
cfg = ev.Config(tau=0.3)
fires = [r for r in rules if any(ev.strength(E[i]["features"], r, cfg) > 0 for i in dev)]
R = len(fires)
sign = np.array([1.0 if r.kind == "favor" else -1.0 for r in fires])

def vlm_rank(i):
    prev = PREV
    return E[i]["top8"] + [p for p in prev if p not in E[i]["top8"]]
PREV = es.prevalence_order(E, dev)

# strengths S[photo][rule]
S = {i: np.array([ev.strength(E[i]["features"], r, cfg) for r in fires]) for i in dev}

def ok(Rk, T, k):
    pos = {p: j for j, p in enumerate(Rk)}
    return max(pos.get(t, 99) for t in T) < k
def metr(rank, ids):
    n = len(ids); out = Counter()
    for i in ids:
        T = set(E[i]["expert"]); Rk = rank[i]
        out["top pick"] += Rk[0] in T
        out["family"] += fam[Rk[0]] in {fam[t] for t in T}
        out["top 3"] += ok(Rk, T, 3)
        out["top 8"] += ok(Rk, T, 8)
    return {k: 100 * v / n for k, v in out.items()}

# ---------- E1: trust signal (no labels used to build it) ----------
rules_only = {i: ev.score(procs, rules, E[i]["features"], E[i]["top8"],
                          ev.Config(prior_weight=0.0, tau=0.3, weight_scale=1.0, tie_break="alpha")) for i in dev}
V = {i: vlm_rank(i) for i in dev}
correct = np.array([V[i][0] in set(E[i]["expert"]) for i in dev])
rank_in_rules = np.array([rules_only[i].index(V[i][0]) for i in dev])        # 0 = rules agree on #1
direct_agree = np.array([E[i]["direct"] == V[i][0] for i in dev])
print("E1  VLM top pick correct overall: %.0f%% (n=%d)" % (100 * correct.mean(), len(dev)))
for lab, m in (("rules also rank it in their top 3", rank_in_rules < 3), ("rules rank it 4th or lower", rank_in_rules >= 3)):
    print("    when %-36s: %4.0f%% correct  (%2.0f%% of photos)" % (lab, 100 * correct[m].mean(), 100 * m.mean()))
for lab, m in (("both VLM channels agree", direct_agree), ("VLM channels disagree", ~direct_agree)):
    print("    when %-36s: %4.0f%% correct  (%2.0f%% of photos)" % (lab, 100 * correct[m].mean(), 100 * m.mean()))
both = direct_agree & (rank_in_rules < 3)
print("    when %-36s: %4.0f%% correct  (%2.0f%% of photos)" % ("VLM agrees with itself AND rules agree", 100 * correct[both].mean(), 100 * both.mean()))
print("    AUROC for spotting VLM errors: rules rank %.2f | VLM self-agreement %.2f" %
      (roc_auc_score(correct, -rank_in_rules), roc_auc_score(correct, direct_agree)))

# ---------- E2: learned weights, 5-fold CV ----------
def rows(i, use_rules):
    T = set(E[i]["expert"]); top8 = E[i]["top8"]
    X, y = [], []
    for p in procs:
        f = np.zeros(9 + 1 + P + (R if use_rules else 0))
        r = top8.index(p) if p in top8 else 8
        f[r] = 1.0                                   # VLM rank one-hot (8 = not in top 8)
        f[9] = 1.0 if E[i]["direct"] == p else 0.0   # VLM direct answer
        f[10 + pidx[p]] = 1.0                        # process base rate
        if use_rules:
            mask = np.array([r_.process == p for r_ in fires])
            f[10 + P:] = np.where(mask, sign * S[i], 0.0)
        X.append(f); y.append(1 if p in T else 0)
    return X, y

def fit_rank(train, test, use_rules, C):
    Xtr, ytr = [], []
    for i in train:
        x, y = rows(i, use_rules); Xtr += x; ytr += y
    m = LogisticRegression(C=C, max_iter=4000)
    m.fit(np.array(Xtr), np.array(ytr))
    out = {}
    for i in test:
        x, _ = rows(i, use_rules)
        pr = m.predict_proba(np.array(x))[:, 1]
        vr = {p: k for k, p in enumerate(V[i])}
        out[i] = sorted(procs, key=lambda p: (-pr[pidx[p]], vr[p]))
    return out, m

rng = random.Random(20261013); ids = list(dev); rng.shuffle(ids)
folds = [ids[k::5] for k in range(5)]
res = {"M0 VLM ranking": {}, "M1 VLM + base rates": {}, "M2 VLM + base rates + rules": {}}
Cs = (0.03, 0.1, 0.3, 1.0)
for k in range(5):
    test = folds[k]; train = [i for j, f in enumerate(folds) if j != k for i in f]
    for i in test: res["M0 VLM ranking"][i] = V[i]
    for name, use in (("M1 VLM + base rates", False), ("M2 VLM + base rates + rules", True)):
        # choose C by inner 4-fold CV on the training photos (average of top-3 and top-8)
        best = None
        for C in Cs:
            sc = []
            for kk in range(4):
                itest = train[kk::4]; itrain = [i for i in train if i not in set(itest)]
                o, _ = fit_rank(itrain, itest, use, C); m = metr(o, itest); sc.append(m["top 3"] + m["top 8"] + m["top pick"])
            if best is None or np.mean(sc) > best[0]: best = (np.mean(sc), C)
        o, _ = fit_rank(train, test, use, best[1]); res[name].update(o)
print("\nE2  5-fold cross-validated on dev (each photo scored by a model that never saw it)")
for name, rk in res.items():
    m = metr(rk, dev)
    print("    %-30s top pick %4.1f%%  family %4.1f%%  top 3 %4.1f%%  top 8 %4.1f%%" % (name, m["top pick"], m["family"], m["top 3"], m["top 8"]))

def boot(a, b, metric, n=4000, seed=20261012):
    def per(rk, i):
        T = set(E[i]["expert"]); Rk = rk[i]
        return {"top pick": Rk[0] in T, "family": fam[Rk[0]] in {fam[t] for t in T}, "top 3": ok(Rk, T, 3), "top 8": ok(Rk, T, 8)}[metric]
    d = np.array([per(res[a], i) - per(res[b], i) for i in dev], dtype=float)
    r = np.random.default_rng(seed); bs = np.sort([d[r.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return 100 * d.mean(), 100 * bs[int(.025 * n)], 100 * bs[int(.975 * n) - 1]
for a, b in (("M2 VLM + base rates + rules", "M0 VLM ranking"), ("M2 VLM + base rates + rules", "M1 VLM + base rates")):
    print("    %s vs %s:" % (a.split()[0], b.split()[0]), "  ".join("%s %+.1f (%+.1f to %+.1f)" % ((m,) + boot(a, b, m)) for m in ("top pick", "family", "top 3", "top 8")))
