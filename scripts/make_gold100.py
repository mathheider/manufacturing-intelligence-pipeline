# -*- coding: utf-8 -*-
"""GOLD-100 selection (protocol in ../GOLD100.md). Deterministic; declared
before any gold-set scoring. Selection uses ONLY image-intrinsic criteria
(QC acceptance + resolution) and label coverage -- never model performance.

Rule:
  0. Curated non-open-source images (the 14 seed archetypes) join FIRST when
     their files exist under images_curated/; they displace the lowest-ranked
     unprotected picks. (2026-09-01: files not yet present.)
  1. Eligible = QC-accepted pool photos (the 635). Rank = min(width,height)
     desc, tie-break path.
  2. Coverage pass: processes ascending by support; each gets its 5
     best-ranked eligible photos (dedup).
  3. Fill to 100 by global rank.
Outputs images_pool/gold100_manifest.json + selection audit.
"""
from __future__ import annotations
import json, collections
from pathlib import Path

MIP = Path(__file__).resolve().parent.parent
man = json.loads((MIP/"images_pool"/"images_manifest.json").read_text(encoding="utf-8"))["images"]
qc  = json.loads((MIP/"outputs"/"qc_report.json").read_text(encoding="utf-8"))["results"]
accepted = {r["path"] for r in qc if r.get("accept")}
pool = [e for e in man if e["path"] in accepted and (MIP/e["path"]).exists()]
pool.sort(key=lambda e: (-min(e["width"], e["height"]), e["path"]))
rank = {e["path"]: i for i, e in enumerate(pool)}

support = collections.Counter()
for e in pool:
    for p in e["expert_processes"]: support[p] += 1

TARGET, PER_PROC = 100, 5
chosen, protected = [], set()
seen = set()
for proc in sorted(support, key=lambda p: (support[p], p)):          # rare first
    got = sum(1 for e in chosen if proc in e["expert_processes"])
    for e in pool:
        if got >= PER_PROC: break
        if e["path"] in seen or proc not in e["expert_processes"]: continue
        chosen.append(e); seen.add(e["path"]); protected.add(e["path"]); got += 1
for e in pool:                                                        # fill
    if len(chosen) >= TARGET: break
    if e["path"] not in seen:
        chosen.append(e); seen.add(e["path"])
chosen = chosen[:TARGET]

cov = collections.Counter(); cats = collections.Counter()
for e in chosen:
    cats[e["category"]] += 1
    for p in e["expert_processes"]: cov[p] += 1
audit = {
    "rule": "QC-accepted; rank=min(w,h) desc; 5-per-process coverage pass (rare first); fill by rank; curated seeds join first when files exist (not yet present 2026-09-01)",
    "n": len(chosen), "protected": len(protected),
    "process_coverage": dict(sorted(cov.items())),
    "category_counts": dict(sorted(cats.items())),
    "min_resolution_selected": min(min(e["width"],e["height"]) for e in chosen),
    "images": [{"path": e["path"], "category": e["category"],
                "expert_processes": e["expert_processes"],
                "rank": rank[e["path"]],
                "protected": e["path"] in protected} for e in chosen],
}
out = MIP/"images_pool"/"gold100_manifest.json"
out.write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(f"wrote {out.name}: n={len(chosen)} protected={len(protected)} "
      f"min_res={audit['min_resolution_selected']}")
print("process coverage:", dict(sorted(cov.items(), key=lambda x: x[1])))
