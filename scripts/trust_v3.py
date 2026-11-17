# -*- coding: utf-8 -*-
"""Learned trust per perceived feature: cross-validated and frozen on the 535 dev photos. Zero API calls.

Score of process p on one photo:
    a[slot of p in the VLM's top 8, or "not in top 8"] + d * [p is the VLM's direct answer]
    + sum over feature groups g of  t_g * sum over rules r for p in group g of  direction_r * weight_r * strength_r
t_g >= 0 is ONE trust number per perceived feature (11) plus one for rules that combine features. The hand-written rule
weights and directions are kept, so a rule can only act the way it is written. There are no per-process base rates, so
the model cannot memorize part types. Logistic loss on (photo, process) pairs, L2 penalty chosen by inner CV.

Dev results come from two 5-fold cross-validations (every photo is scored by a model that never saw it):
  random folds            new photos of part types seen in training
  leave-part-type-out     whole part types held out of training (per-rule weights fail this test)
Models compared: M0 the VLM's own ranking | MV learned VLM signals only, no rules | M4 one trust weight per rule
(stated direction only) | M5 one trust weight per feature (the model that is frozen).

    python scripts/trust_v3.py                   # dev CV + freeze M5 on all dev photos   (needs: pip install numpy scipy)
    python scripts/trust_v3.py --test            # score Gold-100 ONCE with the frozen weights (standard library only)
    python scripts/trust_v3.py --test --labels gold100_labels.json   # same, with per-photo labels {photo_id: [processes]}

Visible-mark answers (outputs/signatures_v3.json) are NOT used: the frozen model reads the 11 archived bench-v2 features.
A model that uses visible marks is a new model and must be frozen under a new name.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

MIP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MIP / "src"))
sys.path.insert(0, str(MIP / "scripts"))
from mip import evidence as ev  # noqa: E402
import eval_soft_v3 as es       # noqa: E402

TAU = 0.5                                   # evidence.Config default, not tuned
LAMBDAS = (0.3, 1.0, 3.0, 10.0, 30.0)
SEED_PHOTOS, SEED_TYPES, SEED_BOOT = 20261013, 7, 20261012
FROZEN = MIP / "outputs" / "trust_v3_frozen.json"
FROZEN_SHA = MIP / "outputs" / "trust_v3_frozen.sha256"
CV_OUT = MIP / "outputs" / "trust_v3_dev_cv.json"
TEST_OUT = MIP / "outputs" / "trust_v3_test.json"
MEASURES = ("top pick correct", "right family", "all correct in top 3", "all correct in top 8")
NO_SIGNATURES = MIP / "outputs" / "__no_signature_reads__.json"     # a path that does not exist: no visible-mark merge
NV = 10                                     # 9 VLM rank slots + the direct answer


def sha256(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def group_of(rule) -> str:
    fs = sorted({c.feature for c in rule.when})
    return fs[0] if len(fs) == 1 else "combination of features"


def load():
    E, fam, _ = es.load(NO_SIGNATURES)
    procs = sorted({p for v in E.values() for p in v["top8"]} | set(fam))
    _, _, rules = ev.load_rules(MIP / "data" / "rules_v3.json")
    return E, fam, procs, rules


def prevalence(E, ids, procs):
    c = Counter(p for i in ids for p in E[i]["expert"])
    return sorted(procs, key=lambda p: (-c[p], p))


def ranked(procs, score, vlm_order):
    pos = {p: k for k, p in enumerate(vlm_order)}
    return sorted(procs, key=lambda p: (-round(score[p], 9), pos[p]))


def per_photo(R, truth, fam):
    pos = {p: k for k, p in enumerate(R)}
    need = max(pos.get(t, 99) for t in truth)
    return (float(R[0] in truth), float(fam[R[0]] in {fam[t] for t in truth}), float(need < 3), float(need < 8))


# ----------------------------------------------------------------------------- fitting (numpy + scipy)
def build(E, procs, rules, dev):
    import numpy as np
    cfg = ev.Config(tau=TAU)
    fires = [r for r in rules if any(ev.strength(E[i]["features"], r, cfg) > 0 for i in dev)]
    groups = sorted({group_of(r) for r in fires})
    pidx = {p: k for k, p in enumerate(procs)}
    sign = np.array([1.0 if r.kind == "favor" else -1.0 for r in fires])
    wt = np.array([r.weight for r in fires])
    rp = np.array([pidx[r.process] for r in fires])
    rg = np.array([groups.index(group_of(r)) for r in fires])
    X = {}
    for i in dev:
        s = np.array([ev.strength(E[i]["features"], r, cfg) for r in fires])
        base = np.zeros((len(procs), NV))
        for p in procs:
            k = pidx[p]
            base[k, E[i]["top8"].index(p) if p in E[i]["top8"] else 8] = 1.0
            base[k, 9] = 1.0 if E[i]["direct"] == p else 0.0
        per_rule = np.zeros((len(procs), len(fires)))
        per_rule[rp, np.arange(len(fires))] = sign * s
        per_feat = np.zeros((len(procs), len(groups)))
        np.add.at(per_feat, (rp, rg), sign * wt * s)
        y = np.array([1.0 if p in set(E[i]["expert"]) else 0.0 for p in procs])
        X[i] = {"MV": (base, y), "M4": (np.hstack([base, per_rule]), y), "M5": (np.hstack([base, per_feat]), y)}
    return X, fires, groups


def fit(X, train, model, lam):
    import numpy as np
    from scipy.optimize import minimize
    A = np.vstack([X[i][model][0] for i in train])
    y = np.concatenate([X[i][model][1] for i in train])
    d = A.shape[1]

    def f(w):
        z = A @ w[:-1] + w[-1]
        q = 1.0 / (1.0 + np.exp(-z))
        loss = (np.logaddexp(0.0, z) - y * z).sum() + 0.5 * lam * (w[:-1] ** 2).sum()
        return loss, np.append(A.T @ (q - y) + lam * w[:-1], (q - y).sum())

    bounds = [(None, None)] * NV + [(0.0, None)] * (d - NV) + [(None, None)]   # rules act only as written
    return minimize(f, np.zeros(d + 1), jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 3000}).x


def rank_np(X, i, model, w, procs, order):
    s = X[i][model][0] @ w[:-1]
    return ranked(procs, {p: float(s[k]) for k, p in enumerate(procs)}, order)


def choose_lambda(X, E, procs, fam, train, model):
    import numpy as np
    best = None
    for lam in LAMBDAS:
        sc = []
        for kk in range(4):
            hold = set(train[kk::4])
            fit_ids = [i for i in train if i not in hold]
            w, order = fit(X, fit_ids, model, lam), prevalence(E, fit_ids, procs)
            sc.append(np.mean([sum(per_photo(rank_np(X, i, model, w, procs, E[i]["top8"] + [p for p in order if p not in E[i]["top8"]]),
                                             set(E[i]["expert"]), fam)[k] for k in (0, 2, 3)) for i in hold]))
        if best is None or np.mean(sc) > best[0]:
            best = (float(np.mean(sc)), lam)
    return best[1]


def cross_validate(X, E, procs, fam, folds):
    import numpy as np
    res, lams = defaultdict(dict), defaultdict(list)
    for k, test in enumerate(folds):
        train = [i for j, f in enumerate(folds) if j != k for i in f]
        order = prevalence(E, train, procs)
        for i in test:
            res["M0"][i] = per_photo(E[i]["top8"] + [p for p in order if p not in E[i]["top8"]], set(E[i]["expert"]), fam)
        for model in ("MV", "M4", "M5"):
            lam = choose_lambda(X, E, procs, fam, train, model)
            w = fit(X, train, model, lam)
            lams[model].append(lam)
            for i in test:
                res[model][i] = per_photo(rank_np(X, i, model, w, procs, E[i]["top8"] + [p for p in order if p not in E[i]["top8"]]),
                                          set(E[i]["expert"]), fam)
    ids = [i for f in folds for i in f]
    table = {m: {MEASURES[c]: round(100 * float(np.mean([res[m][i][c] for i in ids])), 1) for c in range(4)} for m in res}
    rng = np.random.default_rng(SEED_BOOT)
    gains = {}
    for a, b in (("M5", "M0"), ("M5", "MV"), ("M4", "M0"), ("MV", "M0")):
        d = np.array([np.subtract(res[a][i], res[b][i]) for i in ids])
        bs = np.sort(np.array([d[rng.integers(0, len(ids), len(ids))].mean(axis=0) for _ in range(4000)]), axis=0)
        gains[f"{a} vs {b}"] = {MEASURES[c]: {"gain_points": round(100 * float(d.mean(axis=0)[c]), 1),
                                             "range95": [round(100 * float(bs[100, c]), 1), round(100 * float(bs[3899, c]), 1)]}
                                for c in range(4)}
    return table, gains, dict(lams)


def main_fit(a) -> int:
    import numpy as np
    if FROZEN.exists() and not a.refreeze:
        print(f"{FROZEN.name} already exists (sha256 {sha256(FROZEN)[:16]}...). Use --refreeze only if you mean to replace it.")
        return 1
    E, fam, procs, rules = load()
    dev = sorted(i for i, v in E.items() if v["split"] == "dev")
    X, fires, groups = build(E, procs, rules, dev)
    ids = list(dev)
    random.Random(SEED_PHOTOS).shuffle(ids)
    cats = sorted({E[i]["category"] for i in dev})
    random.Random(SEED_TYPES).shuffle(cats)
    schemes = {"random 5-fold (new photos, part types seen in training)": [ids[k::5] for k in range(5)],
               "leave-part-type-out 5-fold (part types never seen in training)":
                   [[i for i in dev if E[i]["category"] in set(cats[k::5])] for k in range(5)]}
    names = {"M0": "VLM alone (its own ranking)", "MV": "VLM signals only, learned (no rules)",
             "M4": "trust per rule, stated direction (no base rates)", "M5": "trust per feature (frozen model)"}
    report = {"_meta": {"what": "dev-only cross-validation; Gold-100 not read", "n_dev": len(dev), "tau": TAU,
                        "lambdas": LAMBDAS, "rules_v3_sha256": sha256(MIP / "data" / "rules_v3.json"),
                        "evidence_sha256": sha256(MIP / "outputs" / "evidence_v2pool.json"), "models": names},
              "schemes": {}}
    print("All numbers are % of dev photos, each scored by a model that never saw it.")
    for label, folds in schemes.items():
        table, gains, lams = cross_validate(X, E, procs, fam, folds)
        report["schemes"][label] = {"percent": table, "gains_vs": gains, "lambda_per_fold": lams}
        print(f"\n{label}")
        for m in ("M0", "MV", "M4", "M5"):
            print(f"  {names[m]:48s} " + "  ".join(f"{k} {v:4.1f}%" for k, v in table[m].items()))
        for g in ("M5 vs M0", "M5 vs MV", "M4 vs M0"):
            print(f"  {g}: " + "  ".join(f"{k} {v['gain_points']:+.1f} ({v['range95'][0]:+.1f} to {v['range95'][1]:+.1f})"
                                         for k, v in gains[g].items()))
    CV_OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")

    # freeze M5 on all dev photos: lambda by the same inner 4-fold procedure
    lam = choose_lambda(X, E, procs, fam, ids, "M5")
    w = fit(X, ids, "M5", lam)
    trust = {g: round(float(w[NV + k]), 6) for k, g in enumerate(groups)}
    frozen = {
        "_meta": {"model": "M5 trust per perceived feature", "frozen_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "fit_on": f"all {len(dev)} dev photos (Gold-100 not read)", "lambda": lam, "tau": TAU,
                  "rules_v3_sha256": sha256(MIP / "data" / "rules_v3.json"),
                  "evidence_sha256": sha256(MIP / "outputs" / "evidence_v2pool.json"),
                  "visible_mark_reads": "not used",
                  "protocol": "score Gold-100 once with --test; a per-photo label file may replace the category-level sets"},
        "rank_slot_weights": [round(float(x), 6) for x in w[:9]],
        "direct_answer_weight": round(float(w[9]), 6),
        "trust": trust,
        "intercept": round(float(w[-1]), 6),
        "rules": [{"id": r.id, "group": group_of(r)} for r in fires],
        "tie_break_prevalence_order": prevalence(E, dev, procs),
    }
    FROZEN.write_text(json.dumps(frozen, indent=1), encoding="utf-8")
    h = sha256(FROZEN)
    FROZEN_SHA.write_text(f"{h}  {FROZEN.name}\n", encoding="utf-8")
    print(f"\nFrozen M5 (lambda {lam}) -> {FROZEN.relative_to(MIP)}  sha256 {h}")
    print("Learned trust per feature (0 = ignored):")
    for g, t in sorted(trust.items(), key=lambda kv: -kv[1]):
        print(f"  {t:5.2f}  {g}")
    return 0


# ----------------------------------------------------------------------------- scoring (standard library only)
def score_frozen(F, E, i, procs, rules_by_id):
    cfg = ev.Config(tau=F["_meta"]["tau"])
    s = {}
    for p in procs:
        slot = E[i]["top8"].index(p) if p in E[i]["top8"] else 8
        s[p] = F["rank_slot_weights"][slot] + (F["direct_answer_weight"] if E[i]["direct"] == p else 0.0)
    for spec in F["rules"]:
        r = rules_by_id[spec["id"]]
        st = ev.strength(E[i]["features"], r, cfg)
        if st > 0:
            s[r.process] += F["trust"][spec["group"]] * (1.0 if r.kind == "favor" else -1.0) * r.weight * st
    return s


def main_test(a) -> int:
    if not FROZEN.exists():
        print("No frozen model yet: run  python scripts/trust_v3.py  first.")
        return 1
    h = sha256(FROZEN)
    if FROZEN_SHA.exists() and FROZEN_SHA.read_text(encoding="utf-8").split()[0] != h:
        print("WARNING: trust_v3_frozen.json does not match trust_v3_frozen.sha256; it was changed after freezing.")
    if TEST_OUT.exists():
        saved = json.loads(TEST_OUT.read_text(encoding="utf-8"))
        if not a.force:
            print(f"Gold-100 was already scored with this frozen model on {saved['_meta']['scored_utc']}. The protocol allows "
                  f"one scoring; see {TEST_OUT.name}. Use --force to re-print the saved result (nothing is re-scored).")
            return 1
        print(json.dumps(saved, indent=1))
        return 0
    F = json.loads(FROZEN.read_text(encoding="utf-8"))
    E, fam, procs, rules = load()
    if sha256(MIP / "data" / "rules_v3.json") != F["_meta"]["rules_v3_sha256"]:
        print("WARNING: rules_v3.json changed since freezing; the frozen weights refer to the old rule base.")
    rules_by_id = {r.id: r for r in rules}
    test = sorted(i for i, v in E.items() if v["split"] == "test")
    labels = json.loads(Path(a.labels).read_text(encoding="utf-8")) if a.labels else None
    truth = {i: set(labels[i]) if labels else set(E[i]["expert"]) for i in test}
    order = F["tie_break_prevalence_order"]
    vlm = {i: E[i]["top8"] + [p for p in order if p not in E[i]["top8"]] for i in test}
    m0 = {i: per_photo(vlm[i], truth[i], fam) for i in test}
    m5 = {i: per_photo(ranked(procs, score_frozen(F, E, i, procs, rules_by_id), vlm[i]), truth[i], fam) for i in test}
    n = len(test)
    pct = {name: {MEASURES[c]: round(100 * sum(d[i][c] for i in test) / n, 1) for c in range(4)}
           for name, d in (("VLM alone", m0), ("trust per feature (frozen)", m5))}
    rng = random.Random(SEED_BOOT)
    diffs = [[m5[i][c] - m0[i][c] for c in range(4)] for i in test]
    boots = []
    for _ in range(4000):
        smp = [diffs[rng.randrange(n)] for _ in range(n)]
        boots.append([sum(x[c] for x in smp) / n for c in range(4)])
    gains = {}
    for c in range(4):
        col = sorted(b[c] for b in boots)
        gains[MEASURES[c]] = {"gain_points": round(100 * sum(x[c] for x in diffs) / n, 1),
                              "range95": [round(100 * col[100], 1), round(100 * col[3899], 1)]}
    out = {"_meta": {"scored_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "n_photos": n,
                     "frozen_sha256": h, "labels": a.labels or "category-level expert sets (evidence_v2pool.json)"},
           "percent": pct, "gain_vs_vlm": gains}
    TEST_OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"Gold-100 ({n} photos), frozen model sha256 {h[:16]}..., all numbers are % of photos")
    for name, row in pct.items():
        print(f"  {name:28s} " + "  ".join(f"{k} {v:4.1f}%" for k, v in row.items()))
    print("  gain vs VLM (points): " + "  ".join(f"{k} {v['gain_points']:+.1f} ({v['range95'][0]:+.1f} to {v['range95'][1]:+.1f})"
                                               for k, v in gains.items()))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="score Gold-100 once with the frozen weights")
    ap.add_argument("--labels", default=None, help="per-photo labels for --test: JSON {photo_id: [processes]}")
    ap.add_argument("--force", action="store_true", help="with --test: re-print the saved scoring (never re-scores)")
    ap.add_argument("--refreeze", action="store_true", help="replace an existing frozen model")
    a = ap.parse_args(argv)
    return main_test(a) if a.test else main_fit(a)


if __name__ == "__main__":
    sys.exit(main())
