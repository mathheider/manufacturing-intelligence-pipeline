"""Random QC sample contact sheet (goal 7 acceptance, CLAUDE_CODE_GOALS.md).

Writes outputs/qc_sample_sheet.html: a seeded-random sample of QC-accepted
images (default 30) with category, QC reason, license, and attribution, for
Mathias to eyeball label quality before the committee does.

    python scripts/contact_sheet.py [--n 30] [--seed 20260701]
"""
from __future__ import annotations

import argparse
import html
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
MANIFEST = HERE / "images_pool" / "images_manifest.json"
QC_REPORT = HERE / "outputs" / "qc_report.json"
OUT = HERE / "outputs" / "qc_sample_sheet.html"

CARD = """<div class="card">
  <img src="{src}" loading="lazy">
  <div class="meta">
    <b>{cat}</b> — {name}<br>
    expert: {procs}<br>
    QC: {reason}<br>
    <span class="lic">{license} — {attribution}</span>
  </div>
</div>"""

PAGE = """<!doctype html><meta charset="utf-8">
<title>MIP QC sample sheet</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 16px; background:#fafafa; }}
 .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px,1fr)); gap: 12px; }}
 .card {{ background:#fff; border:1px solid #ddd; border-radius:6px; overflow:hidden; }}
 .card img {{ width:100%; height:200px; object-fit:cover; display:block; }}
 .meta {{ padding:8px; font-size:12px; line-height:1.5; }}
 .lic {{ color:#777; }}
</style>
<h1>MIP real-image QC sample — {n} random accepted images (seed {seed})</h1>
<p>{total_accepted} accepted total, prompt {pv}. Regenerate:
<code>python scripts/contact_sheet.py --n {n} --seed {seed}</code></p>
<div class="grid">{cards}</div>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260701)
    args = ap.parse_args()

    manifest = {e["path"]: e for e in
                json.loads(MANIFEST.read_text(encoding="utf-8"))["images"]}
    rep = json.loads(QC_REPORT.read_text(encoding="utf-8"))
    accepted = [r for r in rep["results"]
                if r.get("accept") and (HERE / r["path"]).exists()]
    random.Random(args.seed).shuffle(accepted)
    sample = sorted(accepted[:args.n], key=lambda r: (r["category"], r["path"]))

    cards = []
    for r in sample:
        m = manifest.get(r["path"], {})
        cards.append(CARD.format(
            src=html.escape(f"../{r['path']}"),
            cat=html.escape(r["category"]),
            name=html.escape(Path(r["path"]).name),
            procs=html.escape(", ".join(m.get("expert_processes", []))),
            reason=html.escape(r.get("reason", "")),
            license=html.escape(m.get("license", "?")),
            attribution=html.escape(m.get("attribution", "?"))))

    OUT.write_text(PAGE.format(n=len(sample), seed=args.seed,
                               total_accepted=len(accepted),
                               pv=rep.get("prompt_version", "?"),
                               cards="\n".join(cards)), encoding="utf-8")
    print(f"wrote {OUT} ({len(sample)} images)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
