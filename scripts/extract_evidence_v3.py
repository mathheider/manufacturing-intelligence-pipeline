# -*- coding: utf-8 -*-
"""Extract per-photo perception evidence from the archived bench-v2 responses.

Zero API calls: reads outputs/vlm_cache/bench-gemini-2.5-flash-lite/ (extracted
from outputs/vlm_cache_archives/v2_cache_20260805.tar.gz) and maps photos to
cache keys through outputs/image_sha24_v2pool.json (sha256[:24] of each photo,
the key prefix used by vlm/cache.py).

For every photo and every one of the 11 features it collects ALL archived reads:
the Greedy pass (one read of every feature) plus every Hybrid consistency
sample. It also keeps the VLM's single process answer and its ranked top-8.

Writes outputs/evidence_v2pool.json:
  {photo_id: {category, expert, hard, split, direct, top8,
              features: {name: [parsed values...]}}}
Split: Gold-100 photos (images_pool/gold100_manifest.json) = "test", the rest = "dev".

    python scripts/extract_evidence_v3.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

MIP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MIP / "src"))

from mip.rules import RuleSet                      # noqa: E402
from mip.vlm.real import match_process, parse, parse_topk  # noqa: E402

CACHE = MIP / "outputs" / "vlm_cache" / "bench-gemini-2.5-flash-lite"
PAT = re.compile(r"^([0-9a-f]{24})_(greedy|hybrid|direct|topk)-(.+)_(\d\d)_bench-v2\.json$")


def gold_ids() -> set:
    g = json.loads((MIP / "images_pool" / "gold100_manifest.json").read_text(encoding="utf-8"))
    items = g.get("images") or g.get("selected") or g.get("set") or g
    ids = set()
    for it in items if isinstance(items, list) else []:
        p = it["path"] if isinstance(it, dict) else str(it)
        ids.add(Path(p).stem)
    return ids


def main() -> int:
    rules = RuleSet.load(None)
    ftype = {n: f.type for n, f in rules.features.items()}
    shamap = json.loads((MIP / "outputs" / "image_sha24_v2pool.json").read_text(encoding="utf-8"))["images"]
    by_sha = {v["sha24"]: k for k, v in shamap.items()}

    recs = {}
    for line in (MIP / "outputs" / "records_real_v2.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("method") == "Direct":
            recs[r["id"]] = r
        if r.get("method") == "Direct@k":
            recs.setdefault("__topk__" + r["id"], r)
    gold = gold_ids()

    raw = defaultdict(lambda: defaultdict(list))   # id -> channel -> [(idx, text)]
    n_files = 0
    for fp in CACHE.iterdir():
        m = PAT.match(fp.name)
        if not m:
            continue
        sha, pipe, chan, idx = m.groups()
        pid = by_sha.get(sha)
        if pid is None:
            continue
        n_files += 1
        txt = json.loads(fp.read_text(encoding="utf-8"))["raw"]
        raw[pid][f"{pipe}:{chan}"].append((int(idx), txt))

    out, mismatch = {}, 0
    for pid, r in recs.items():
        if pid.startswith("__topk__"):
            continue
        ch = raw[pid]
        feats = {}
        for f, t in ftype.items():
            vals = []
            for pipe in ("greedy", "hybrid"):
                for _, txt in sorted(ch.get(f"{pipe}:{f}", [])):
                    vals.append(parse(txt, t))
            feats[f] = vals
        direct = match_process(sorted(ch["direct:__process__"])[0][1])
        top8 = parse_topk(sorted(ch["topk:__process_top8__"])[0][1], 8)
        rk = recs.get("__topk__" + pid)
        if rk and rk.get("ranked") != top8:
            mismatch += 1
        out[pid] = {"category": r["category"], "expert": sorted(r["expert"]), "hard": r["hard"],
                    "split": "test" if pid in gold else "dev",
                    "direct": direct, "top8": top8, "features": feats}
    meta = {"source": "outputs/vlm_cache/bench-gemini-2.5-flash-lite (bench-v2 archive, no API calls)",
            "n_photos": len(out), "n_cache_files_used": n_files,
            "n_dev": sum(v["split"] == "dev" for v in out.values()),
            "n_test": sum(v["split"] == "test" for v in out.values()),
            "top8_mismatch_vs_records": mismatch}
    (MIP / "outputs" / "evidence_v2pool.json").write_text(
        json.dumps({"_meta": meta, "photos": out}), encoding="utf-8")
    print(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
