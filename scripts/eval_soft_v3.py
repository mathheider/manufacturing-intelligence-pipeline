# -*- coding: utf-8 -*-
"""Evaluate soft-evidence rule sets on the archived bench-v2 responses (zero API calls).

Every method produces a full ranking of the 17 processes per photo. Reported
as plain percentages of photos (higher is better):
  top pick correct       the #1 process is one of the correct processes
  right family           the #1 process is in one of the correct families
  all correct in top 3   every correct process is in the top 3
  all correct in top 8   every correct process is in the top 8
Settings are chosen on dev by the average of "all correct in top k" over
k = 1..8 (the selection rule frozen on 2026-10-12); only the reporting changed.

Baselines: no-look prevalence ranking (fit on dev labels only), the VLM's own
top-8 (Direct@8, unranked processes appended in prevalence order), and the
frozen hard pipelines (Greedy, Hybrid) from outputs/records_real_v2.jsonl.

Dev = 535 photos, test = Gold-100 (held out). Configs are chosen on dev only;
``--test`` scores the frozen choice on test once.

    python scripts/eval_soft_v3.py                # dev
    python scripts/eval_soft_v3.py --test         # frozen config on Gold-100
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

MIP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MIP / "src"))

from mip import evidence as ev  # noqa: E402

OBS = {"hollow": "visible", "constant_cross_section": "visible", "axisymmetric": "visible",
       "undercut_present": "visible", "rib_count": "visible",
       "max_dimension_mm": "estimable", "projected_area_cm2": "estimable",
       "wall_thickness_mm": "not_visible", "draft_angle_deg": "not_visible",
       "production_volume": "not_visible", "continuous_fiber_required": "not_visible"}
KS = (1, 2, 3, 5, 8)


def load(sig_path=None):
    E = json.loads((MIP / "outputs" / "evidence_v2pool.json").read_text(encoding="utf-8"))["photos"]
    sig_path = Path(sig_path) if sig_path else MIP / "outputs" / "signatures_v3.json"
    if sig_path.exists():                       # merge visible-signature reads when the pass has run
        S = json.loads(sig_path.read_text(encoding="utf-8"))["photos"]
        for pid, reads in S.items():
            if pid in E:
                E[pid]["features"].update(reads)
    MV = json.loads((MIP / "outputs" / "metrics_v3.json").read_text(encoding="utf-8"))
    fam = MV["family_map"]
    hard = {}
    for line in (MIP / "outputs" / "records_real_v2.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("method") in ("Hybrid", "Greedy"):
            hard[(r["method"], r["id"])] = r["ranked"]
    return E, fam, hard


def metrics(rankings: dict, E: dict, fam: dict, ids) -> dict:
    n = 0
    acc = Counter()
    for i in ids:
        R = rankings[i]
        T = set(E[i]["expert"])
        n += 1
        pos = {p: k for k, p in enumerate(R)}
        need = max((pos.get(t, 99) for t in T)) + 1          # rank that contains all
        for k in range(1, 9):
            acc[f"c{k}"] += need <= k
        acc["hit1"] += bool(R) and R[0] in T
        acc["fam1"] += bool(R) and fam[R[0]] in {fam[t] for t in T}
        acc["cov"] += min(need, 18)
    out = {f"cont@{k}": acc[f"c{k}"] / n for k in KS}
    out["AUC"] = sum(acc[f"c{k}"] for k in range(1, 9)) / (8 * n)
    out["hit@1"] = acc["hit1"] / n
    out["fam@1"] = acc["fam1"] / n
    out["cov"] = acc["cov"] / n
    out["n"] = n
    return out


def prevalence_order(E, ids):
    c = Counter(p for i in ids for p in E[i]["expert"])
    procs = sorted({p for v in E.values() for p in v["top8"]} | set(c))
    return sorted(procs, key=lambda p: (-c[p], p))


def baselines(E, fam, hard, ids, prev):
    out = {}
    out["no-look prevalence"] = {i: list(prev) for i in ids}
    out["VLM top-8 (Direct@8)"] = {i: E[i]["top8"] + [p for p in prev if p not in E[i]["top8"]] for i in ids}
    for m in ("Greedy", "Hybrid"):
        out[f"{m} hard v2 (frozen)"] = {i: hard[(m, i)] for i in ids}
    return out


def run(processes, rules, E, ids, cfg):
    return {i: ev.score(processes, rules, E[i]["features"], E[i]["top8"], cfg) for i in ids}


def grid(processes, rules, E, fam, ids, prior_ws=(0.0, 1.0, 2.0, 4.0, 8.0), taus=(0.3, 0.5, 0.7),
         scales=(0.5, 1.0, 2.0), obs_sets=None):
    obs_sets = obs_sets or [
        {"visible": 1.0, "signature": 1.0, "estimable": 0.5, "not_visible": 0.25},
        {"visible": 1.0, "signature": 1.0, "estimable": 1.0, "not_visible": 1.0},
        {"visible": 1.0, "signature": 1.0, "estimable": 0.5, "not_visible": 0.0},
        {"visible": 1.0, "signature": 1.0, "estimable": 1.0, "not_visible": 0.5}]
    best = None
    for pw, tau, sc, ob in itertools.product(prior_ws, taus, scales, obs_sets):
        cfg = ev.Config(prior_weight=pw, tau=tau, weight_scale=sc, obs_w=ob,
                        tie_break=("alpha" if pw == 0 else "vlm"))
        m = metrics(run(processes, rules, E, ids, cfg), E, fam, ids)
        key = (round(m["AUC"], 4), round(m["hit@1"], 4))
        if best is None or key > best[0]:
            best = (key, cfg, m)
    return best


def fmt_row(name, m):
    """Plain percentages, out of 100 photos (higher is better)."""
    return (f"{name:34s}  top pick correct {100*m['hit@1']:3.0f}%  right family {100*m['fam@1']:3.0f}%"
            f"  all correct in top 3 {100*m['cont@3']:3.0f}%  all correct in top 8 {100*m['cont@8']:3.0f}%")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules-v3", default=str(MIP / "data" / "rules_v3.json"))
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--signatures", default=None, help="signature reads (default outputs/signatures_v3.json if present)")
    ap.add_argument("--validity", default=str(MIP / "outputs" / "rule_validity_v3.json"))
    a = ap.parse_args(argv)
    E, fam, hard = load(a.signatures)
    dev = sorted(i for i, v in E.items() if v["split"] == "dev")
    test = sorted(i for i, v in E.items() if v["split"] == "test")
    prev = prevalence_order(E, dev)
    processes = sorted({p for v in E.values() for p in v["top8"]} | set(fam))
    v2soft = ev.from_hard_rules(MIP / "data" / "rules.json", weight=3.0, observability=OBS)
    v3 = ev.load_rules(a.rules_v3)[2] if Path(a.rules_v3).exists() else None
    hard_ids = [i for i in dev if E[i]["hard"]]

    report = {"split": "test" if a.test else "dev", "baselines": {}, "variants": {}}
    print("All numbers are % of photos (higher is better).")
    for subset_name, sub in (("ALL", dev), ("HARD", hard_ids)):
        print(f"--- DEV {subset_name} (n={len(sub)})")
        for name, rk in baselines(E, fam, hard, sub, prev).items():
            m = metrics(rk, E, fam, sub)
            report["baselines"][f"{subset_name}:{name}"] = m
            print(fmt_row(name, m))

    variants = {"soft v2 (92 rules, no veto)": v2soft}
    if v3:
        variants["soft v3 (expanded)"] = v3
        if Path(a.validity).exists():
            ok = {r["id"] for r in json.loads(Path(a.validity).read_text(encoding="utf-8"))["rules"] if r["validated"]}
            variants[f"soft v3 validated ({len(ok)})"] = [r for r in v3 if r.id in ok]
    chosen = {}
    for name, rules in variants.items():
        for prior_mode, pws in (("rules only", (0.0,)), ("rules + VLM prior", (0.5, 1.0, 2.0, 4.0, 8.0))):
            key, cfg, m = grid(processes, rules, E, fam, dev, prior_ws=pws)
            mh = metrics(run(processes, rules, E, hard_ids, cfg), E, fam, hard_ids)
            label = f"{name} | {prior_mode}"
            chosen[label] = cfg
            report["variants"][label] = {"cfg": cfg.__dict__, "ALL": m, "HARD": mh, "n_rules": len(rules)}
            print(fmt_row(label[:34], m) + f"   [HARD AUC {mh['AUC']:.3f} hit@1 {mh['hit@1']:.3f}] cfg pw={cfg.prior_weight} tau={cfg.tau} sc={cfg.weight_scale} obs={cfg.obs_w['not_visible']}")

    if a.test:
        test_hard = [i for i in test if E[i]["hard"]]
        report["test"] = {}
        print(f"\n=== TEST (Gold-100, n={len(test)}; HARD n={len(test_hard)}) with dev-frozen configs")
        for subset_name, sub in (("ALL", test), ("HARD", test_hard)):
            for name, rk in baselines(E, fam, hard, sub, prev).items():
                m = metrics(rk, E, fam, sub)
                report["test"][f"{subset_name}:{name}"] = m
                print(f"[{subset_name}] " + fmt_row(name, m))
            for label, cfg in chosen.items():
                rules = variants[label.split(" | ")[0]]
                m = metrics(run(processes, rules, E, sub, cfg), E, fam, sub)
                report["test"][f"{subset_name}:{label}"] = m
                print(f"[{subset_name}] " + fmt_row(label[:34], m))
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
