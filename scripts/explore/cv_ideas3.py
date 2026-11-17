# -*- coding: utf-8 -*-
"""Exploratory, DEV ONLY, zero API calls. Variants that cannot memorize part types:
  M4  VLM signals + per-rule trust (stated direction only), NO process base rates
  M5  VLM signals + one trust weight per perceived feature (hand rule weights kept), NO base rates  (~22 parameters)"""
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
cfg = ev.Config(tau=0.5)
fires = [r for r in rules if any(ev.strength(E[i]["features"], r, cfg) > 0 for i in dev)]
R = len(fires); sign = np.array([1.0 if r.kind == "favor" else -1.0 for r in fires]); hw = np.array([r.weight for r in fires])
rproc = np.array([pidx[r.process] for r in fires])
def gkey(r):
    fs = sorted({c.feature for c in r.when}); return fs[0] if len(fs) == 1 else "combination of features"
G = sorted({gkey(r) for r in fires}); gidx = {g: k for k, g in enumerate(G)}; rg = np.array([gidx[gkey(r)] for r in fires])
S = {i: np.array([ev.strength(E[i]["features"], r, cfg) for r in fires]) for i in dev}
NV = 10
def feats(i, mode):
    top8 = E[i]["top8"]; ext = {"M4": R, "M5": len(G)}[mode]; X = np.zeros((P, NV + ext))
    for p in procs:
        k = pidx[p]; X[k, top8.index(p) if p in top8 else 8] = 1.0; X[k, 9] = 1.0 if E[i]["direct"] == p else 0.0
    for j in range(R):
        v = sign[j] * S[i][j]
        if mode == "M4": X[rproc[j], NV + j] = v
        else: X[rproc[j], NV + rg[j]] += v * hw[j]
    y = np.array([1.0 if p in set(E[i]["expert"]) else 0.0 for p in procs]); return X, y
F = {(i, m): feats(i, m) for i in dev for m in ("M4", "M5")}
def fit(train, mode, lam):
    X = np.vstack([F[(i, mode)][0] for i in train]); y = np.concatenate([F[(i, mode)][1] for i in train]); d = X.shape[1]
    def f(w):
        z = X @ w[:-1] + w[-1]; l = np.logaddexp(0, z) - y * z; g = X.T @ (1 / (1 + np.exp(-z)) - y)
        return l.sum() + 0.5 * lam * (w[:-1] ** 2).sum(), np.append(g + lam * w[:-1], (1 / (1 + np.exp(-z)) - y).sum())
    b = [(None, None)] * NV + [(0.0, None)] * (d - NV) + [(None, None)]
    return minimize(f, np.zeros(d + 1), jac=True, method="L-BFGS-B", bounds=b, options={"maxiter": 3000}).x
def prev_order(ids):
    c = Counter(p for i in ids for p in E[i]["expert"]); return sorted(procs, key=lambda p: (-c[p], p))
def rank_with(w, i, mode, prev):
    s = F[(i, mode)][0] @ w[:-1]; vr = {p: k for k, p in enumerate(E[i]["top8"] + [p for p in prev if p not in E[i]["top8"]])}
    return sorted(procs, key=lambda p: (-round(s[pidx[p]], 9), vr[p]))
def ok(Rk, T, k):
    pos = {p: j for j, p in enumerate(Rk)}; return max(pos.get(t, 99) for t in T) < k
def per(Rk, i):
    T = set(E[i]["expert"]); return np.array([Rk[0] in T, fam[Rk[0]] in {fam[t] for t in T}, ok(Rk, T, 3), ok(Rk, T, 8)], dtype=float)
LAMS = (0.3, 1.0, 3.0, 10.0, 30.0)
def run(folds, label):
    res = defaultdict(dict); W = defaultdict(list)
    for k, test in enumerate(folds):
        train = [i for j, f in enumerate(folds) if j != k for i in f]; prev = prev_order(train)
        for i in test: res["M0 VLM ranking"][i] = per(E[i]["top8"] + [p for p in prev if p not in E[i]["top8"]], i)
        for name, mode in (("M4 per-rule trust, no base rates", "M4"), ("M5 per-feature trust, no base rates", "M5")):
            best = None
            for lam in LAMS:
                sc = []
                for kk in range(4):
                    it = set(train[kk::4]); itr = [i for i in train if i not in it]; pv = prev_order(itr); w = fit(itr, mode, lam)
                    sc.append(np.mean([per(rank_with(w, i, mode, pv), i)[[0, 2, 3]].sum() for i in it]))
                if best is None or np.mean(sc) > best[0]: best = (np.mean(sc), lam)
            w = fit(train, mode, best[1]); W[name].append(w)
            for i in test: res[name][i] = per(rank_with(w, i, mode, prev), i)
    print(f"\n{label}")
    ids = [i for f in folds for i in f]; rng = np.random.default_rng(20261012)
    for name, d in res.items():
        print("  %-36s top pick %4.1f%%  family %4.1f%%  top 3 %4.1f%%  top 8 %4.1f%%" % ((name,) + tuple(100 * np.mean([d[i] for i in ids], axis=0))))
    for a in ("M4 per-rule trust, no base rates", "M5 per-feature trust, no base rates"):
        d = np.array([res[a][i] - res["M0 VLM ranking"][i] for i in ids]); n = len(ids)
        bs = np.sort(np.array([d[rng.integers(0, n, n)].mean(axis=0) for _ in range(4000)]), axis=0)
        print("  %s vs VLM: " % a.split(" ")[0] + "  ".join("%s %+.1f (%+.1f to %+.1f)" % (nm, 100 * d.mean(axis=0)[c], 100 * bs[100, c], 100 * bs[3899, c]) for c, nm in enumerate(("top pick", "family", "top 3", "top 8"))))
    return W
rng = random.Random(20261013); ids = list(dev); rng.shuffle(ids)
W1 = run([ids[k::5] for k in range(5)], "RANDOM 5-FOLD (new photos, part types seen in training)")
cats = sorted({E[i]["category"] for i in dev}); random.Random(7).shuffle(cats); cf = [set(cats[k::5]) for k in range(5)]
W2 = run([[i for i in dev if E[i]["category"] in cf[k]] for k in range(5)], "LEAVE-PART-TYPE-OUT 5-FOLD (unseen part types)")
w = np.mean([x[NV:-1] for x in W2["M5 per-feature trust, no base rates"]], axis=0)
print("\nM5 learned trust per perceived feature (leave-part-type-out, mean over folds; 0 = ignored):")
for k in np.argsort(-w): print(f"  {w[k]:5.2f}  {G[k]}")
