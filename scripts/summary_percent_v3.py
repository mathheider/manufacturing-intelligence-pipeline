# -*- coding: utf-8 -*-
"""Plain-percentage summary of the v3 results (what the write-up reports).

Every number is "out of 100 photos" (higher is better):
  Top pick correct       the #1 ranked process is one of the correct processes
  Right family           the #1 ranked process is in one of the correct process families
  All correct in top 3   every correct process for that photo is in the top 3
  All correct in top 8   every correct process for that photo is in the top 8

Gains are in percentage points against the VLM alone, with a 95% range from
4,000 paired bootstrap resamples of photos. Uses the configs frozen on dev
(outputs/soft_v3_frozen_dev_configs.json); zero API calls.

    python scripts/summary_percent_v3.py
"""
import json, random, sys
from pathlib import Path
MIP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MIP / "src")); sys.path.insert(0, str(MIP / "scripts"))
from mip import evidence as ev
import eval_soft_v3 as E3

COLS = [("hit1", "Top pick correct"), ("fam1", "Right family"), ("c3", "All correct in top 3"), ("c8", "All correct in top 8")]


def r0(x):
    """Round half up, so the gains shown equal the difference of the percentages shown."""
    return int(x + 0.5 + 1e-9) if x >= 0 else -int(-x + 0.5 + 1e-9)


def per_photo(R, T, fam):
    pos = {p: k for k, p in enumerate(R)}
    need = max(pos.get(t, 99) for t in T) + 1
    if not R:                                   # hard rules can eliminate every process
        return {"hit1": 0.0, "fam1": 0.0, "c3": 0.0, "c8": 0.0}
    return {"hit1": float(R[0] in T), "fam1": float(fam[R[0]] in {fam[t] for t in T}),
            "c3": float(need <= 3), "c8": float(need <= 8)}


def main():
    E, fam, hard = E3.load()
    frozen = json.loads((MIP / "outputs" / "soft_v3_frozen_dev_configs.json").read_text(encoding="utf-8"))["variants"]
    _, _, R3 = ev.load_rules(MIP / "data" / "rules_v3.json")
    v2 = ev.from_hard_rules(MIP / "data" / "rules.json", 3.0, E3.OBS)
    ok = {r["id"] for r in json.loads((MIP / "outputs" / "rule_validity_v3.json").read_text(encoding="utf-8"))["rules"] if r["validated"]}
    R3v = [r for r in R3 if r.id in ok]
    dev = sorted(i for i, v in E.items() if v["split"] == "dev")
    prev = E3.prevalence_order(E, dev)
    procs = sorted(fam)

    def cfg_of(prefix, mode):
        lab = next(k for k in frozen if k.startswith(prefix) and mode in k)
        return ev.Config(**frozen[lab]["cfg"])

    methods = [
        ("Guess the most common processes (ignores the photo)", lambda i: list(prev)),
        ("VLM alone (its own top-8 ranking)", lambda i: E[i]["top8"] + [p for p in prev if p not in E[i]["top8"]]),
        ("Current Hybrid (rules delete processes)", lambda i: hard[("Hybrid", i)]),
        ("Original 92 rules as evidence + VLM", lambda i, c=cfg_of("soft v2", "VLM"): ev.score(procs, v2, E[i]["features"], E[i]["top8"], c)),
        ("All 564 rules + VLM", lambda i, c=cfg_of("soft v3 (expanded)", "VLM"): ev.score(procs, R3, E[i]["features"], E[i]["top8"], c)),
        (f"{len(ok)} checked rules + VLM", lambda i, c=cfg_of("soft v3 validated", "VLM"): ev.score(procs, R3v, E[i]["features"], E[i]["top8"], c)),
        (f"{len(ok)} checked rules alone (no VLM)", lambda i, c=cfg_of("soft v3 validated", "rules only"): ev.score(procs, R3v, E[i]["features"], E[i]["top8"], c)),
    ]
    out = {"metrics": dict(COLS), "tables": {}, "gains_vs_vlm_points": {}}
    for split in ("test", "dev"):
        for sub in ("all", "hard"):
            ids = sorted(i for i, v in E.items() if v["split"] == split and (sub == "all" or v["hard"]))
            per = {}
            for name, fn in methods:
                per[name] = {i: per_photo(fn(i), set(E[i]["expert"]), fam) for i in ids}
            key = f"{split}:{sub}"
            out["tables"][key] = {name: {c: round(100 * sum(per[name][i][c] for i in ids) / len(ids), 1) for c, _ in COLS} | {"n": len(ids)}
                                  for name, _ in methods}
            base = per[methods[1][0]]
            out["gains_vs_vlm_points"][key] = {}
            for name, _ in methods[3:6]:
                g = {}
                for c, _ in COLS:
                    d = [per[name][i][c] - base[i][c] for i in ids]
                    rnd = random.Random(20261012); bs = []
                    for _ in range(4000):
                        bs.append(sum(d[rnd.randrange(len(d))] for _ in d) / len(d))
                    bs.sort()
                    g[c] = {"gain": round(100 * sum(d) / len(d), 1), "range95": [round(100 * bs[100], 1), round(100 * bs[3899], 1)]}
                out["gains_vs_vlm_points"][key][name] = g
    (MIP / "outputs" / "soft_v3_summary_percent.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for key, tab in out["tables"].items():
        print(f"\n== {key} (n={next(iter(tab.values()))['n']})")
        print(f"{'':52s}" + "".join(f"{lab:>22s}" for _, lab in COLS))
        for name, row in tab.items():
            print(f"{name:52s}" + "".join(f"{r0(row[c]):>21d}%" for c, _ in COLS))
        vlm = tab["VLM alone (its own top-8 ranking)"]
        for name, g in out["gains_vs_vlm_points"][key].items():
            print(f"  vs VLM alone (points), {name:38s}" + "".join(
                f"  {r0(tab[name][c]) - r0(vlm[c]):+d} ({r0(g[c]['range95'][0]):+d} to {r0(g[c]['range95'][1]):+d})" for c, _ in COLS))


if __name__ == "__main__":
    main()
