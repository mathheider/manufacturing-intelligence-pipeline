"""Fetch open-licensed part images for the MIP real-image benchmark.

Hardened port of mip_agent_prototype/fetch_images.py (goal 5, CLAUDE_CODE_GOALS.md):
  * Sources: Wikimedia Commons API (primary) + Openverse API (fallback).
  * License filter: CC0 / CC BY / CC BY-SA / Public domain only.
  * Quality gates: min resolution (short side, default 512px; downloads are
    768px-wide thumbs so height is the usual constraint), average-hash dedupe
    across the whole pool, mime/extension checks.
  * Resumable: re-running tops each category up to --per-category without
    re-downloading; manifest records source, license, attribution, size, hash.
  * Polite: contact User-Agent, 429 backoff, small sleep between downloads.

Usage:
    python scripts/fetch_images.py --per-category 35              # full pool
    python scripts/fetch_images.py --per-category 3 --only fuel_tank,hood
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import time
from pathlib import Path

import requests
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent))
from image_categories import CATEGORIES  # noqa: E402

HERE = Path(__file__).resolve().parent.parent          # mip/
POOL = HERE / "images_pool"
MANIFEST = POOL / "images_manifest.json"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
OPENVERSE_API = "https://api.openverse.org/v1/images/"
# Wikimedia and Openverse ask for a contact in the User-Agent. Set MIP_CONTACT
# to your own email address or project URL before fetching.
CONTACT = os.environ.get(
    "MIP_CONTACT", "https://github.com/mathheider/manufacturing-intelligence-pipeline"
)
HEADERS = {"User-Agent": f"MIP-benchmark/0.3 (research; {CONTACT})"}
OK_LICENSE = re.compile(r"(CC0|CC.?BY(.?SA)?|Public domain|PD)", re.IGNORECASE)


def strip_html(t: str) -> str:
    return re.sub(r"<[^>]+>", "", t or "").strip()


def get(url, **kw):
    for attempt in range(5):
        r = requests.get(url, headers=HEADERS, timeout=45, **kw)
        if r.status_code == 429 or r.status_code >= 500:   # rate limit / transient
            time.sleep(2.0 * (attempt + 1))
            continue
        r.raise_for_status()
        return r
    raise RuntimeError(f"rate-limited after retries: {url}")


def search_commons(term: str, want: int) -> list[dict]:
    """Paged Commons search (generator continuation) until ``want`` license-OK
    candidates or results run dry (max ~400 scanned per term)."""
    out: list[dict] = []
    offset = 0
    while len(out) < want and offset < 400:
        params = {"action": "query", "format": "json", "generator": "search",
                  "gsrsearch": f"{term} filetype:bitmap", "gsrnamespace": 6,
                  "gsrlimit": 50, "gsroffset": offset, "prop": "imageinfo",
                  "iiprop": "url|extmetadata|mime", "iiurlwidth": 768}
        data = get(COMMONS_API, params=params).json()
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            break
        for p in pages.values():
            info = (p.get("imageinfo") or [{}])[0]
            if not info.get("mime", "").startswith("image/") or info.get("mime") == "image/svg+xml":
                continue
            meta = info.get("extmetadata", {})
            lic = strip_html(meta.get("LicenseShortName", {}).get("value", ""))
            if not OK_LICENSE.search(lic):
                continue
            out.append({"dl": info.get("thumburl") or info.get("url"),
                        "source": "wikimedia_commons",
                        "url": info.get("descriptionurl", ""),
                        "license": lic,
                        "attribution": strip_html(meta.get("Artist", {}).get("value", "")) or "unknown",
                        "title": p.get("title", "")})
        cont = data.get("continue", {}).get("gsroffset")
        if cont is None:
            break
        offset = cont
    return out


def search_openverse(term: str, want: int) -> list[dict]:
    """Paged Openverse search until ``want`` candidates or results run dry."""
    out: list[dict] = []
    page = 1
    while len(out) < want and page <= 5:
        params = {"q": term, "license": "cc0,by,by-sa", "page_size": 20,
                  "page": page, "filter_dead": "true"}
        try:
            results = get(OPENVERSE_API, params=params).json().get("results", [])
        except Exception:
            break
        if not results:
            break
        for r in results:
            if (r.get("filetype") or "").lower() == "svg":
                continue
            out.append({"dl": r.get("url"),
                        "source": "openverse",
                        "url": r.get("foreign_landing_url") or r.get("url", ""),
                        "license": f"CC {r.get('license','?').upper()} {r.get('license_version','')}".strip(),
                        "attribution": r.get("creator") or "unknown",
                        "title": r.get("title") or ""})
        page += 1
    return out


def ahash(img: Image.Image) -> str:
    """64-bit average hash for near-duplicate detection."""
    g = img.convert("L").resize((8, 8), Image.LANCZOS)
    px = list(g.getdata())
    avg = sum(px) / 64
    return "".join("1" if p > avg else "0" for p in px)


def hamming(a: str, b: str) -> int:
    return sum(c1 != c2 for c1, c2 in zip(a, b))


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {"images": []}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=35)
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--min-res", type=int, default=512,
                    help="minimum short-side pixels")
    ap.add_argument("--dedupe-dist", type=int, default=4,
                    help="max hamming distance considered a duplicate")
    args = ap.parse_args()

    only = set(filter(None, args.only.split(",")))
    POOL.mkdir(exist_ok=True)
    manifest = load_manifest()
    have_paths = {m["path"] for m in manifest["images"]}
    have_urls = {m["url"] for m in manifest["images"]}
    have_hashes = [m["ahash"] for m in manifest["images"] if m.get("ahash")]

    for cat, spec in CATEGORIES.items():
        if only and cat not in only:
            continue
        cdir = POOL / cat
        cdir.mkdir(exist_ok=True)
        # count only entries whose file is still on disk, so slots freed by a
        # QC --apply (rejects moved away) are replenished on the next run
        n_have = sum(1 for m in manifest["images"]
                     if m["category"] == cat and (HERE / m["path"]).exists())
        need = args.per_category - n_have
        if need <= 0:
            print(f"{cat}: already {n_have}, skip")
            continue
        used_idx = [int(Path(m["path"]).stem.rsplit("_", 1)[1])
                    for m in manifest["images"] if m["category"] == cat]
        next_idx = (max(used_idx) + 1) if used_idx else 0

        candidates: list[dict] = []
        for term in spec["search_terms"]:
            candidates += search_commons(term, want=need * 3)
        if len(candidates) < need * 2:
            for term in spec["search_terms"]:
                candidates += search_openverse(term, want=need * 2)

        added = 0
        for cand in candidates:
            if added >= need:
                break
            if not cand["dl"] or cand["url"] in have_urls:
                continue
            try:
                blob = get(cand["dl"]).content
                img = Image.open(io.BytesIO(blob))
                img.load()
            except Exception as e:
                print(f"  skip (download/decode): {str(e)[:60]}")
                continue
            if min(img.size) < args.min_res:
                continue
            h = ahash(img)
            if any(hamming(h, old) <= args.dedupe_dist for old in have_hashes):
                continue
            idx = next_idx + added
            rel = f"images_pool/{cat}/{cat}_{idx:03d}.jpg"
            dest = HERE / rel
            img.convert("RGB").save(dest, "JPEG", quality=90)
            manifest["images"].append({
                "path": rel, "category": cat,
                "expert_processes": sorted(spec["expert_processes"]),
                "source": cand["source"], "url": cand["url"],
                "license": cand["license"], "attribution": cand["attribution"],
                "width": img.size[0], "height": img.size[1], "ahash": h,
            })
            have_urls.add(cand["url"])
            have_hashes.append(h)
            added += 1
            time.sleep(0.4)
        print(f"{cat}: +{added} (now {n_have + added}/{args.per_category})")
        MANIFEST.write_text(json.dumps(manifest, indent=1))

    print(f"\nTOTAL in pool: {len(manifest['images'])}")


if __name__ == "__main__":
    main()
