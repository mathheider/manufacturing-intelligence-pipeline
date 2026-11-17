# -*- coding: utf-8 -*-
"""Perception pass for the 28 visible-signature features of rules_v3.json.

One call per photo asks ALL signature questions at once and requests a JSON
object back (a batched questionnaire), so a full pass over the 635-photo
bench-v2 pool is ~635 x N calls instead of 635 x 28 x N. Same model and
temperature as bench-v2 (gemini-2.5-flash-lite, 0.7) so the evidence is
comparable with the archived reads.

Every raw reply is cached under outputs/vlm_cache/sig-v3-<model>/ keyed by
image sha256[:24] + sample index (same convention as vlm/cache.py): the run is
resumable, auditable, and a re-run is free. Parsed reads go to
outputs/signatures_v3.json, which eval_soft_v3.py merges automatically.

Order: dev photos first, Gold-100 (test) last, so a --limit pilot never touches
the held-out set. Needs GEMINI_API_KEY (read from the environment or the repo's
.env) and `pip install "mip[gemini]"`.

    python scripts/perceive_signatures_v3.py --limit 60            # dev pilot
    python scripts/perceive_signatures_v3.py                       # full pass, 3 samples
    python scripts/perceive_signatures_v3.py --backend mock        # plumbing test, no API
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

MIP = Path(__file__).resolve().parent.parent
PROMPT_VERSION = "sig-v3"


def load_env() -> None:
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        return
    for p in (MIP.parent / ".env", MIP / ".env"):
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def build_prompt(sig: dict) -> str:
    lines = [
        "You are an expert in plastics and composites manufacturing, inspecting a photo "
        "of a single part for physical marks left by how it was made.",
        "Answer every question below about THIS photo only. Use 'unknown' when the "
        "relevant surface is not visible in the photo. Do not guess the process; "
        "report only what you can see.",
        "Return ONLY a JSON object mapping each key to \"yes\", \"no\" or \"unknown\".",
        "",
    ]
    for k, spec in sig.items():
        q = spec["question"].replace(" Answer yes or no.", "")
        lines.append(f"{k}: {q}")
    return "\n".join(lines)


def parse_reply(raw: str, keys) -> dict:
    out = {k: None for k in keys}
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return out
    try:
        obj = json.loads(m.group())
    except json.JSONDecodeError:
        return out
    for k in keys:
        v = str(obj.get(k, "")).strip().lower()
        out[k] = True if v in ("yes", "true", "y") else False if v in ("no", "false", "n") else None
    return out


class GeminiSig:
    def __init__(self, model: str, temperature: float):
        import google.generativeai as genai          # pip install "mip[gemini]"
        load_env()
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("set GEMINI_API_KEY (or put it in the repo .env)")
        genai.configure(api_key=key)
        self.genai = genai
        self.model = genai.GenerativeModel(model)
        self.temperature = temperature

    def call(self, image_path: Path, prompt: str) -> str:
        from PIL import Image
        cfg = self.genai.types.GenerationConfig(temperature=self.temperature, max_output_tokens=1024)
        last = None
        for attempt in range(6):                     # 429/5xx backoff; never cache a failure
            try:
                img = Image.open(str(image_path)); img.load()
                resp = self.model.generate_content([prompt, img], generation_config=cfg)
                try:
                    return (resp.text or "").strip()
                except ValueError:
                    return "{}"
            except Exception as e:                   # noqa: BLE001
                last = e
                time.sleep(3.0 * (attempt + 1))
        raise RuntimeError(f"Gemini call failed after retries: {last}") from last


class MockSig:
    """Plumbing test only: deterministic pseudo-random answers, never cached as real."""
    def __init__(self, keys):
        self.keys = list(keys)

    def call(self, image_path: Path, prompt: str) -> str:
        r = random.Random(str(image_path) + prompt[-40:])
        return json.dumps({k: r.choice(["yes", "no", "no", "unknown"]) for k in self.keys})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="gemini", choices=["gemini", "mock"])
    ap.add_argument("--model", default="gemini-2.5-flash-lite")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--n", type=int, default=3, help="samples per photo")
    ap.add_argument("--limit", type=int, default=0, help="photos to process (dev first)")
    ap.add_argument("--rules", default=str(MIP / "data" / "rules_v3.json"))
    a = ap.parse_args(argv)

    rules = json.loads(Path(a.rules).read_text(encoding="utf-8"))
    sig = {k: v for k, v in rules["features"].items() if k.startswith("sig_")}
    prompt = build_prompt(sig)
    shamap = json.loads((MIP / "outputs" / "image_sha24_v2pool.json").read_text(encoding="utf-8"))["images"]
    gold = {Path(e["path"]).stem for e in json.loads(
        (MIP / "images_pool" / "gold100_manifest.json").read_text(encoding="utf-8"))["images"]}
    order = sorted(shamap, key=lambda i: (i in gold, i))        # dev first, test last
    if a.limit:
        order = order[:a.limit]

    tag = "mock" if a.backend == "mock" else a.model
    cache = MIP / "outputs" / "vlm_cache" / f"sig-v3-{tag}"
    cache.mkdir(parents=True, exist_ok=True)
    backend = MockSig(sig) if a.backend == "mock" else GeminiSig(a.model, a.temperature)
    out_path = MIP / "outputs" / ("signatures_v3_mock.json" if a.backend == "mock" else "signatures_v3.json")
    results = json.loads(out_path.read_text(encoding="utf-8"))["photos"] if out_path.exists() else {}

    calls = hits = 0
    for n_done, pid in enumerate(order, 1):
        img = MIP / shamap[pid]["path"]
        sha = shamap[pid]["sha24"]
        if a.backend != "mock" and img.exists() and hashlib.sha256(img.read_bytes()).hexdigest()[:24] != sha:
            raise RuntimeError(f"{pid}: image bytes changed since bench-v2 (sha mismatch)")
        reads = {k: [] for k in sig}
        for s in range(a.n):
            fp = cache / f"{sha}_sig_{s:02d}_{PROMPT_VERSION}.json"
            if fp.exists():
                raw = json.loads(fp.read_text(encoding="utf-8"))["raw"]
                hits += 1
            else:
                if a.backend != "mock" and not img.exists():
                    raise FileNotFoundError(img)
                raw = backend.call(img, prompt)
                calls += 1
                fp.write_text(json.dumps({"raw": raw, "prompt_version": PROMPT_VERSION,
                                          "model": tag, "prompt": prompt}), encoding="utf-8")
            for k, v in parse_reply(raw, sig).items():
                reads[k].append(v)
        results[pid] = reads
        if n_done % 25 == 0 or n_done == len(order):
            out_path.write_text(json.dumps({"_meta": {"prompt_version": PROMPT_VERSION, "model": tag,
                                                      "samples_per_photo": a.n, "n_photos": len(results)},
                                            "photos": results}), encoding="utf-8")
            print(f"{n_done}/{len(order)} photos | new calls {calls} | cache hits {hits}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
