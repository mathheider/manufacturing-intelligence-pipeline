"""Manifest schema check (goal 5 acceptance, CLAUDE_CODE_GOALS.md).

Validates every entry in images_pool/images_manifest.json:
  * required fields present with the right types;
  * category exists in image_categories.CATEGORIES and expert_processes
    matches the category's set exactly;
  * license matches the fetcher's allow-list (CC0 / CC BY / CC BY-SA / PD);
  * ahash is a 64-char bitstring; url is http(s); attribution non-empty;
  * resolution >= --min-res on the short side;
  * no duplicate paths or source URLs; near-duplicate ahash pairs reported.

Exit code 0 = valid (warnings allowed), 1 = schema errors.

    python scripts/check_manifest.py [--min-res 512] [--strict]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from image_categories import CATEGORIES  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
MANIFEST = HERE / "images_pool" / "images_manifest.json"
OK_LICENSE = re.compile(r"(CC0|CC.?BY(.?SA)?|Public domain|PD)", re.IGNORECASE)

REQUIRED = {"path": str, "category": str, "expert_processes": list,
            "source": str, "url": str, "license": str, "attribution": str,
            "width": int, "height": int, "ahash": str}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-res", type=int, default=512)
    ap.add_argument("--dedupe-dist", type=int, default=4)
    ap.add_argument("--strict", action="store_true",
                    help="warnings also fail the check")
    args = ap.parse_args()

    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))["images"]
    errors, warnings = [], []
    seen_paths, seen_urls = set(), set()

    for i, e in enumerate(entries):
        tag = f"[{i}] {e.get('path', '?')}"
        for field, ftype in REQUIRED.items():
            if field not in e:
                errors.append(f"{tag}: missing field {field!r}")
            elif not isinstance(e[field], ftype):
                errors.append(f"{tag}: {field} should be {ftype.__name__}, "
                              f"got {type(e[field]).__name__}")
        if e.get("path") in seen_paths:
            errors.append(f"{tag}: duplicate path")
        seen_paths.add(e.get("path"))
        if e.get("url") in seen_urls:
            errors.append(f"{tag}: duplicate source url")
        seen_urls.add(e.get("url"))

        cat = e.get("category")
        if cat not in CATEGORIES:
            errors.append(f"{tag}: unknown category {cat!r}")
        elif sorted(e.get("expert_processes", [])) != sorted(CATEGORIES[cat]["expert_processes"]):
            errors.append(f"{tag}: expert_processes drifted from category set")
        if not OK_LICENSE.search(e.get("license", "")):
            errors.append(f"{tag}: license {e.get('license')!r} not in allow-list")
        if not str(e.get("url", "")).startswith("http"):
            errors.append(f"{tag}: url is not http(s)")
        if not e.get("attribution"):
            errors.append(f"{tag}: empty attribution")
        h = e.get("ahash", "")
        if not (len(h) == 64 and set(h) <= {"0", "1"}):
            errors.append(f"{tag}: ahash is not a 64-bit bitstring")
        if isinstance(e.get("width"), int) and isinstance(e.get("height"), int):
            if min(e["width"], e["height"]) < args.min_res:
                warnings.append(f"{tag}: short side {min(e['width'], e['height'])} "
                                f"< {args.min_res}")
        if not (HERE / e.get("path", "")).exists():
            warnings.append(f"{tag}: file missing on disk (QC-rejected?)")

    # near-duplicate scan (report only)
    hashes = [(e["path"], e["ahash"]) for e in entries if e.get("ahash")]
    dupes = 0
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            d = sum(c1 != c2 for c1, c2 in zip(hashes[i][1], hashes[j][1]))
            if d <= args.dedupe_dist:
                dupes += 1
                warnings.append(f"near-dup (hamming {d}): {hashes[i][0]} ~ {hashes[j][0]}")

    for w in warnings:
        print(f"WARN  {w}")
    for e in errors:
        print(f"ERROR {e}")
    print(f"\n{len(entries)} entries: {len(errors)} errors, {len(warnings)} warnings "
          f"({dupes} near-duplicate pairs)")
    if errors or (args.strict and warnings):
        return 1
    print("manifest schema: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
