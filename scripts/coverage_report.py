"""Process-coverage table (goal 6 acceptance, CLAUDE_CODE_GOALS.md).

Prints, for each of the 17 processes: the categories whose expert set contains
it, the fetched image count, and the QC-accepted count. Flags any process
covered by fewer than 2 categories or short of images.

    python scripts/coverage_report.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from image_categories import CATEGORIES  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
MANIFEST = HERE / "images_pool" / "images_manifest.json"
QC_REPORT = HERE / "outputs" / "qc_report.json"
RULES = HERE / "data" / "rules.json"


def main() -> int:
    processes = json.loads(RULES.read_text(encoding="utf-8"))["processes"]

    fetched = defaultdict(int)
    if MANIFEST.exists():
        for e in json.loads(MANIFEST.read_text(encoding="utf-8"))["images"]:
            if (HERE / e["path"]).exists():
                fetched[e["category"]] += 1
    accepted = defaultdict(int)
    if QC_REPORT.exists():
        for r in json.loads(QC_REPORT.read_text(encoding="utf-8"))["results"]:
            if r.get("accept"):
                accepted[r["category"]] += 1

    proc_cats = defaultdict(list)
    for cat, spec in CATEGORIES.items():
        for p in spec["expert_processes"]:
            proc_cats[p].append(cat)

    print(f"{'process':42s} {'cats':>4s} {'fetched':>8s} {'accepted':>9s}  categories")
    print("-" * 110)
    problems = 0
    for p in processes:
        cats = sorted(proc_cats.get(p, []))
        nf = sum(fetched[c] for c in cats)
        na = sum(accepted[c] for c in cats)
        flag = ""
        if len(cats) < 2:
            flag = "  << needs >=2 categories"
            problems += 1
        elif QC_REPORT.exists() and na < 10:
            flag = "  << thin (accepted <10)"
        print(f"{p:42s} {len(cats):4d} {nf:8d} {na:9d}  {', '.join(cats)}{flag}")

    unknown = set().union(*(s["expert_processes"] for s in CATEGORIES.values())) - set(processes)
    if unknown:
        print(f"\nERROR: categories reference unknown processes: {sorted(unknown)}")
        problems += 1

    total_f = sum(fetched.values())
    total_a = sum(accepted.values())
    print("-" * 110)
    print(f"{len(CATEGORIES)} categories, {total_f} images on disk, "
          f"{total_a} QC-accepted")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
