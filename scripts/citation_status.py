"""Proposed-vs-verified citation table (goal 2 acceptance, CLAUDE_CODE_GOALS.md).

Every rule clause whose cite begins with "[VERIFY:" is a Phase-1 proposal that
Mathias must check against the actual source text before it counts as cited.
This prints the full status table and the checklist of clauses to verify.

    python scripts/citation_status.py
"""
from __future__ import annotations

import json
from pathlib import Path

RULES = Path(__file__).resolve().parent.parent / "data" / "rules.json"


def main() -> int:
    raw = json.loads(RULES.read_text(encoding="utf-8"))
    verified, proposed = [], []
    for proc, body in raw["rules"].items():
        for c in body["forbidden_if"]:
            row = (proc, c["feature"], c["op"], c["value"], c["cite"], c["reason"])
            (proposed if c["cite"].startswith("[VERIFY") else verified).append(row)

    print(f"rule clauses: {len(verified) + len(proposed)} total = "
          f"{len(verified)} verified + {len(proposed)} PROPOSED (cite [VERIFY])\n")

    print("== PROPOSED — Mathias must check each against the actual text ==")
    print(f"{'process':40s} {'clause':44s} proposed source")
    print("-" * 130)
    for proc, feat, op, val, cite, reason in proposed:
        clause = f"{feat} {op} {val}"
        src = cite.replace("[VERIFY:", "").rstrip("]").strip()
        print(f"{proc:40s} {clause:44s} {src}")
        print(f"{'':40s}   reason: {reason}")

    print("\n== VERIFIED (transcribed from rules.yaml / LIT_REVIEW.md) ==")
    for proc, feat, op, val, cite, _ in verified:
        print(f"{proc:40s} {feat} {op} {val}   [{cite}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
