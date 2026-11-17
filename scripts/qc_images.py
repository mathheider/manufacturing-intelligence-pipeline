"""VLM screening pass over the real-image pool (goal 7, CLAUDE_CODE_GOALS.md).

One cheap Gemini vision call per image: "does this photo clearly show
<category description>?" Responses are cached on (image sha256, prompt
version), so re-runs cost nothing. Report-only by default; --apply moves
rejects to images_pool/_rejected/.

Prompt v2 (2026-07-01): the question now embeds each category's qc_desc from
image_categories.py, which locks the domain -- the v1 prompt let an aircraft
instrument panel pass as a "dashboard". Bumping PROMPT_VERSION invalidates the
v1 cache on purpose.

Usage:
    python scripts/qc_images.py                 # QC everything, report only
    python scripts/qc_images.py --only fuel_tank --apply
Requires GEMINI_API_KEY in the environment (or ../.env).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import requests

import sys
sys.path.insert(0, str(Path(__file__).parent))
from image_categories import CATEGORIES  # noqa: E402

HERE = Path(__file__).resolve().parent.parent            # mip/
MANIFEST = HERE / "images_pool" / "images_manifest.json"
CACHE = HERE / "outputs" / "vlm_cache" / "qc"
REPORT = HERE / "outputs" / "qc_report.json"
PROMPT_VERSION = "v2"
MODEL = "gemini-2.5-flash"
API = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

PROMPT = ("You are auditing images for a manufacturing benchmark dataset. "
          "Target: {desc}. "
          "Question: does this photograph clearly show that target as the main "
          "subject, photographed as a real physical part (not a diagram, CAD "
          "render, logo, screenshot, toy, or unrelated scene)? Be strict about "
          "the domain: if the description says automotive/car, reject aircraft, "
          "marine, furniture, or architectural lookalikes; a photo of a whole "
          "vehicle only counts if the named part is the clear main subject. "
          'Reply with ONLY minified JSON: {{"accept": true|false, "reason": "<15 words max>"}}')


def api_key() -> str:
    k = os.environ.get("GEMINI_API_KEY")
    if not k:
        env = HERE.parent / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("GEMINI_API_KEY="):
                    k = line.split("=", 1)[1].strip()
    if not k:
        raise SystemExit("GEMINI_API_KEY not set (env or ../.env)")
    return k


def qc_one(key: str, img_path: Path, category: str) -> dict:
    sha = hashlib.sha256(img_path.read_bytes()).hexdigest()[:24]
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE / f"{sha}_{PROMPT_VERSION}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    desc = CATEGORIES.get(category, {}).get(
        "qc_desc", f"a {category.replace('_', ' ')} (an automotive/engineering part)")
    payload = {
        "contents": [{"parts": [
            {"inline_data": {"mime_type": "image/jpeg",
                             "data": base64.b64encode(img_path.read_bytes()).decode()}},
            {"text": PROMPT.format(desc=desc)},
        ]}],
        "generationConfig": {"responseMimeType": "application/json",
                             "temperature": 0.0},
    }
    for attempt in range(5):
        r = requests.post(API, json=payload, timeout=60,
                          headers={"x-goog-api-key": key})
        if r.status_code == 429 or r.status_code >= 500:   # rate limit / transient
            time.sleep(3.0 * (attempt + 1))
            continue
        r.raise_for_status()
        break
    else:
        r.raise_for_status()
    try:
        text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        verdict = json.loads(text)
        verdict = {"accept": bool(verdict.get("accept")),
                   "reason": str(verdict.get("reason", ""))[:120]}
    except Exception as e:
        verdict = {"accept": False, "reason": f"unparseable response: {str(e)[:60]}"}
    verdict["sha"] = sha
    verdict["model"] = MODEL
    cache_file.write_text(json.dumps(verdict))
    return verdict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--limit", type=int, default=0, help="max images this run")
    ap.add_argument("--apply", action="store_true",
                    help="move rejects to images_pool/_rejected/")
    ap.add_argument("--quiet", action="store_true", help="only print the summary")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel QC calls (cache is per-image, thread-safe)")
    args = ap.parse_args()
    only = set(filter(None, args.only.split(",")))
    key = api_key()

    manifest = json.loads(MANIFEST.read_text())
    todo = []
    for entry in manifest["images"]:
        if only and entry["category"] not in only:
            continue
        if args.limit and len(todo) >= args.limit:
            break
        if (HERE / entry["path"]).exists():
            todo.append(entry)

    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            verdicts = list(pool.map(
                lambda e: qc_one(key, HERE / e["path"], e["category"]), todo))
    else:
        verdicts = [qc_one(key, HERE / e["path"], e["category"]) for e in todo]

    report = {"prompt_version": PROMPT_VERSION, "model": MODEL, "results": []}
    n = 0
    for entry, v in zip(todo, verdicts):
        p = HERE / entry["path"]
        report["results"].append({"path": entry["path"],
                                  "category": entry["category"], **v})
        n += 1
        if not args.quiet:
            print(f"{'ACCEPT' if v['accept'] else 'REJECT':6s} {entry['path']}  {v['reason']}")
        if args.apply and not v["accept"]:
            rej = HERE / "images_pool" / "_rejected" / Path(entry["path"]).name
            rej.parent.mkdir(exist_ok=True)
            p.rename(rej)

    acc = sum(1 for r in report["results"] if r["accept"])
    by_cat = defaultdict(lambda: {"accepted": 0, "rejected": 0})
    for r in report["results"]:
        by_cat[r["category"]]["accepted" if r["accept"] else "rejected"] += 1
    report["summary"] = {"checked": n, "accepted": acc, "rejected": n - acc,
                         "by_category": dict(sorted(by_cat.items()))}
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1))
    print(f"\n{acc}/{n} accepted. Report: {REPORT}")
    for cat, c in sorted(by_cat.items()):
        print(f"  {cat:24s} +{c['accepted']:<4d} -{c['rejected']}")


if __name__ == "__main__":
    main()
