# -*- coding: utf-8 -*-
"""Exploratory, DEV ONLY (535 photos), zero API calls. Gold-100 is not read.
Learned rule weights, checked three ways:
  random 5-fold CV (new photos of seen part types) and leave-part-type-out 5-fold CV (unseen part types);
  M2 free rule weights vs M3 sign-constrained (each rule can only act in its stated direction).
Fold-internal everything: prevalence order, base rates, regularization (inner CV)."""
import json, sys, random
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

MIP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MIP / "src")); sys.path.insert(0, str(MIP / "scripts"))
from mip import evidence as ev
import eval_soft_v3 as es

E, fam, _ = es.load(None)
dev = sorted(i for i, v in E.items() if v["split"] == "dev")
procs = sorted({p for v in E.values() for p in v["top8"]} | set(fam)); P = len(procs); pidx = {p: k for k, p in enumerate(procs)}
_, _, rules = ev.load_rules(MIP / "data" / "rules_v3.json")
cfg = ev.Config(tau=0.5)                                   # library default, not the dev-tuned 0.3
fires = [r for r in rules if any(ev.strength(E[i]["features"], r, cfg) > 0 for i in dev)]
R = len(fires); sign = np.array([1.0 if r.kind == "favor" else -1.0 for r in fires])
rproc = np.array([pidx[r.process] for r in fires])
S = {i: np.array([ev.strength(E[i]["features"], r, cfg) for r in fires]) for i in dev}
NV = 9 + 1                                                 # VLM rank one-hot + direct answer
def feats(i, use_rules):
    top8 = E[i]["top8"]; X = np.zeros((P, NV + P + (R if use_rules else 0)))
    for p in procs:
        k = pidx[p]; X[k, top8.index(p) if p in top8 else 8] = 1.0
        X[k, 9] = 1.0 if E[i]["direct"] == p else 0.0; X[k, NV + k] = 1.0
    if use_rules:
        for j in range(R): X[rproc[j], NV + P + j] = sign[j] * S[i][j]
    y = np.array([1.0 if p in set(E[i]["expert"]) else 0.0 for p in procs])
    return X, y
F = {(i, u): feats(i, u) for i in dev for u in (False, True)}

def fit(train, use_rules, lam, constrained):
    X = np.vstack([F[(i, use_rules)][0] for i in train]); y = np.concatenate([F[(i, use_rules)][1] for i in train])
    d = X.shape[1]; pen = np.ones(d); pen[NV:NV + P] = 0.0          # do not shrink base rates
    def f(w):
        z = X @ w[:-1] + w[-1]; l = np.logaddexp(0, z) - y * z
        g = X.T @ (1 / (1 + np.exp(-z)) - y)
        return l.sum() + 0.5 * lam * (pen * w[:-1] ** 2).sum(), np.append(g + lam * pen * w[:-1], (1 / (1 + np.exp(-z)) - y).sum())
    b = [(None, None)] * (d + 1)
    if use_rules and constrained:
        for j in range(NV + P, d): b[j] = (0.0, None)
    w = minimize(f, np.zeros(d + 1), jac=True, method="L-BFGS-B", bounds=b, options={"maxiter": 3000}).x
    return w

def prev_order(ids):
    c = Counter(p for i in ids for p in E[i]["expert"]); return sorted(procs, key=lambda p: (-c[p], p))
def rank_with(w, i, use_rules, prev):
    X, _ = F[(i, use_rules)]; s = X @ w[:-1]
    vr = {p: k for k, p in enumerate(E[i]["top8"] + [p for p in prev if p not in E[i]["top8"]])}
    return sorted(procs, key=lambda p: (-round(s[pidx[p]], 9), vr[p]))
def ok(Rk, T, k):
    pos = {p: j for j, p in enumerate(Rk)}; return max(pos.get(t, 99) for t in T) < k
def per(Rk, i):
    T = set(E[i]["expert"])
    return np.array([Rk[0] in T, fam[Rk[0]] in {fam[t] for t in T}, ok(Rk, T, 3), ok(Rk, T, 8)], dtype=float)
LAMS = (0.3, 1.0, 3.0, 10.0, 30.0)

def run(folds, label):
    res = defaultdict(dict); chosen = defaultdict(list); W = {}
    for k, test in enumerate(folds):
        train = [i for j, f in enumerate(folds) if j != k for i in f]; prev = prev_order(train)
        for i in test:
            res["M0 VLM ranking"][i] = per(E[i]["top8"] + [p for p in prev if p not in E[i]["top8"]], i)
        for name, use, con in (("M1 VLM + base rates", False, False), ("M2 + rules, free weights", True, False), ("M3 + rules, stated direction only", True, True)):
            best = None
            for lam in LAMS:
                sc = []
                for kk in range(4):
                    it = set(train[kk::4]); itr = [i for i in train if i not in it]; pv = prev_order(itr)
                    w = fit(itr, use, lam, con)
                    sc.append(np.mean([per(rank_with(w, i, use, pv), i)[[0, 2, 3]].sum() for i in it]))
                if best is None or np.mean(sc) > best[0]: best = (np.mean(sc), lam)
            chosen[name].append(best[1]); w = fit(train, use, best[1], con); W.setdefault(name, []).append(w)
            for i in test: res[name][i] = per(rank_with(w, i, use, prev), i)
    print(f"\n{label}")
    ids = [i for f in folds for i in f]
    for name, d in res.items():
        m = 100 * np.mean([d[i] for i in ids], axis=0)
        print("  %-34s top pick %4.1f%%  family %4.1f%%  top 3 %4.1f%%  top 8 %4.1f%%" % ((name,) + tuple(m)) + (f"   lambda {chosen[name]}" if name in chosen else ""))
    rng = np.random.default_rng(20261012)
    for a in ("M2 + rules, free weights", "M3 + rules, stated direction only"):
        for b in ("M0 VLM ranking", "M1 VLM + base rates"):
            d = np.array([res[a][i] - res[b][i] for i in ids]); n = len(ids)
            bs = np.sort(np.array([d[rng.integers(0, n, n)].mean(axis=0) for _ in range(4000)]), axis=0)
            print("  %s vs %s: " % (a.split(" ")[0], b.split(" ")[0]) + "  ".join("%s %+.1f (%+.1f to %+.1f)" % (nm, 100 * d.mean(axis=0)[c], 100 * bs[100, c], 100 * bs[3899, c]) for c, nm in enumerate(("top pick", "family", "top 3", "top 8"))))
    return W

rng = random.Random(20261013); ids = list(dev); rng.shuffle(ids)
W1 = run([ids[k::5] for k in range(5)], "RANDOM 5-FOLD (new photos, part types seen in training)")
cats = sorted({E[i]["category"] for i in dev}); random.Random(7).shuffle(cats)
cf = [set(cats[k::5]) for k in range(5)]
W2 = run([[i for i in dev if E[i]["category"] in cf[k]] for k in range(5)], f"LEAVE-PART-TYPE-OUT 5-FOLD ({len(cats)} part types; each fold's types never seen in training)")

# what did the free model learn? sign flips among the larger weights
w = np.mean([x[NV + P:-1] for x in W1["M2 + rules, free weights"]], axis=0)
big = np.abs(w) > 0.25
print(f"\nFree weights (random CV, mean over folds): {big.sum()} rules with |w|>0.25; {int(((w < 0) & big).sum())} of them act AGAINST their stated direction")
wc = np.mean([x[NV + P:-1] for x in W1["M3 + rules, stated direction only"]], axis=0)
order = np.argsort(-wc)
print("Top rules by learned trust (stated direction only):")
for j in order[:12]:
    r = fires[j]; print(f"  {wc[j]:5.2f}  {r.id:7s} {r.kind:8s} {r.process:30s} {r.reason[:60]}")
print(f"Rules kept with weight > 0.05: {(wc > 0.05).sum()} of {R}")
