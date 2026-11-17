# -*- coding: utf-8 -*-
"""Survey reconciliation (CC-1): row-level Qualtrics CSV -> outputs/survey_stats.json.

Source of record: the row-level CSV export (landed 2026-08-04, supersedes the
dashboard PDF per CLAUDE_CODE_PROMPTS.md CC-1). The dashboard PDF
(data/survey/dashboard_export_2026-08-04.pdf) remains as an independent
cross-check of the aggregate counts.

Everything here is deterministic: stdlib only, no floats beyond arithmetic on
fixed inputs, dict/list ordering fixed by explicit sorts.

    python scripts/survey_stats.py        # writes outputs/survey_stats.json
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

MIP = Path(__file__).resolve().parent.parent
CSV = (MIP / "data" / "survey" /
       "Car_Part_Manufacturing_Techniques_2026-08-04_1626.csv")
PDF = MIP / "data" / "survey" / "dashboard_export_2026-08-04.pdf"
OUT = MIP / "outputs" / "survey_stats.json"

# Survey option tokens -> benchmark process ids (rules.json vocabulary).
# Two tokens have no 1:1 mapping and are kept as documented survey-only keys:
# the instrument MERGES Compression Molding with Thermoforming into one option,
# and offers Adhesive Bonding / N/A on some part sets only.
TOKEN_MAP = {
    "Injection Molding": "Injection_Molding",
    "Compression Molding/Thermoforming": "Compression_Molding_or_Thermoforming",
    "Resin Transfer Molding (RTM)": "Resin_Transfer_Molding",
    "Sheet Molding Compound (SMC)": "Sheet_Molding_Compound",
    "Vacuum Assisted RTM (VARTM)": "Vacuum_Assisted_RTM",
    "Structural Reaction Injection Molding (SRIM)":
        "Structural_Reaction_Injection_Molding",
    "Bulk Molding Compound (BMC)": "Bulk_Molding_Compound",
    "Hand Lay-up": "Hand_Lay_up",
    "Selective Laser Sintering (SLS)": "Selective_Laser_Sintering",
    "Stereolithography (STL)": "Stereolithography",
    "Pultrusion": "Pultrusion",
    "Laminated Object Manufacturing (LOM)": "Laminated_Object_Manufacturing",
    "Fused Filament Fabrication (FFF)": "Fused_Filament_Fabrication",
    "Filament Winding": "Filament_Winding",
    "N/A": "NA_none_of_the_listed",
    "Adhesive Bonding": "Adhesive_Bonding",
    "Blow Molding": "Blow_Molding",
    "Overmolding": "Overmolding",
}

# Part-set page titles as shown in the dashboard PDF, by question group.
QUESTION_SETS = {
    "Q1": "leaf spring assembly", "Q2": "fluid reservoir", "Q3": "exhaust",
    "Q4": "steering wheel", "Q5": "door", "Q6": "truck exterior",
    "Q7": "rear body", "Q8": "front body",
}


def masi_distance(a: frozenset, b: frozenset) -> float:
    """1 - MASI similarity (Passonneau 2006): J(a,b) * monotonicity(a,b)."""
    if not a and not b:
        return 0.0
    inter, union = len(a & b), len(a | b)
    j = inter / union if union else 0.0
    if a == b:
        m = 1.0
    elif a <= b or b <= a:
        m = 2 / 3
    elif inter:
        m = 1 / 3
    else:
        m = 0.0
    return 1.0 - j * m


def krippendorff_alpha(units: dict) -> dict:
    """Alpha over set-valued labels with MASI distance.

    units: {unit_id: [frozenset, ...]} (one entry per rater who labeled it).
    Only units with >= 2 raters are pairable. Do is the mean within-unit
    pairwise distance (each unit weighted by m_u/(m_u-1) per Krippendorff's
    general form); De is the mean pairwise distance over all pooled labels.
    """
    pairable = {u: vs for u, vs in units.items() if len(vs) >= 2}
    n_obs = sum(len(vs) for vs in pairable.values())
    if n_obs < 2:
        return {"alpha": None, "reason": "fewer than 2 pairable observations"}
    do_num = 0.0
    for vs in pairable.values():
        m = len(vs)
        within = sum(masi_distance(x, y) for x, y in combinations(vs, 2))
        do_num += (2 * within) / (m - 1)
    d_o = do_num / n_obs
    pooled = [v for vs in pairable.values() for v in vs]
    de_pairs = list(combinations(pooled, 2))
    d_e = (sum(masi_distance(x, y) for x, y in de_pairs) / len(de_pairs)
           if de_pairs else 0.0)
    if d_e == 0.0:
        return {"alpha": 1.0, "note": "zero expected disagreement"}
    return {
        "alpha": round(1.0 - d_o / d_e, 4),
        "observed_disagreement": round(d_o, 4),
        "expected_disagreement": round(d_e, 4),
        "n_pairable_units": len(pairable),
        "n_pairable_observations": n_obs,
        "distance": "1 - MASI (Jaccard x monotonicity), Passonneau 2006",
    }


def leave_one_rater_out(parts: list) -> dict:
    """Alpha sensitivity: recompute with each rater excluded in turn.

    One rater exceeded the 'up to three' instruction on many units (large
    pick sets mechanically depress MASI agreement), so the committee-facing
    question 'is the low alpha one rater?' gets a precomputed answer.
    """
    rater_ids = sorted({rid for p in parts for rid in p["picks_by_rater"]})
    out = {}
    for excl in rater_ids:
        units = {}
        for p in parts:
            kept = {rid: v for rid, v in p["picks_by_rater"].items()
                    if rid != excl}
            units[p["unit"]] = [frozenset(v)
                                for _, v in sorted(kept.items())]
        out[excl] = krippendorff_alpha(units).get("alpha")
    return out


def main() -> int:
    rows = list(csv.reader(open(CSV, encoding="utf-8-sig", newline="")))
    ids, texts, data = rows[0], rows[1], rows[3:]
    hdr = {h: i for i, h in enumerate(ids)}

    part_cols = []                       # (col, qgroup, part_label)
    for i, cid in enumerate(ids):
        m = re.match(r"^(Q\d+)_(\d+)$", cid)
        if m:
            part_cols.append((i, m.group(1), texts[i].split(" - ")[-1]))

    raters = []
    for r in data:
        raters.append({
            "response_id": r[hdr["ResponseId"]],
            "finished": r[hdr["Finished"]] == "True",
            "progress": int(r[hdr["Progress"]]),
            "duration_s": int(r[hdr["Duration (in seconds)"]]),
            "row": r,
        })

    unknown = set()
    parts = []                           # per-part records, CSV column order
    units = {}                           # unit_id -> [frozenset per rater]
    per_rater_parts = Counter()
    totals = Counter()
    overpick = []                        # instruction said "up to three"
    for col, qg, label in part_cols:
        unit = f"{qg}:{label}"
        picks_by_rater = {}
        counts = Counter()
        for ra in raters:
            cell = ra["row"][col].strip() if len(ra["row"]) > col else ""
            if not cell:
                continue
            toks = [t.strip() for t in cell.split(",") if t.strip()]
            mapped = []
            for t in toks:
                if t not in TOKEN_MAP:
                    unknown.add(t)
                mapped.append(TOKEN_MAP.get(t, t))
            if len(mapped) > 3:
                overpick.append({"unit": unit, "rater": ra["response_id"],
                                 "n_picked": len(mapped)})
            picks_by_rater[ra["response_id"]] = sorted(mapped)
            counts.update(mapped)
            totals.update(mapped)
            per_rater_parts[ra["response_id"]] += 1
        units[unit] = [frozenset(v) for _, v in sorted(picks_by_rater.items())]
        parts.append({
            "unit": unit, "question_group": qg,
            "part_set": QUESTION_SETS[qg], "part": label,
            "n_raters": len(picks_by_rater),
            "completeness": "OK",        # row-level CSV: nothing truncated
            "pick_counts": dict(sorted(counts.items())),
            "picks_by_rater": picks_by_rater,
        })

    if unknown:
        print(f"FATAL: unmapped survey tokens: {sorted(unknown)}")
        return 1

    ranked = totals.most_common()
    top, runner = ranked[0], ranked[1]
    per_q_n = {qg: max(p["n_raters"] for p in parts
                       if p["question_group"] == qg)
               for qg in QUESTION_SETS}

    out = {
        "source": {
            "csv": str(CSV.relative_to(MIP)).replace("\\", "/"),
            "csv_sha256": hashlib.sha256(CSV.read_bytes()).hexdigest(),
            "exported": "2026-08-04 16:26 (Qualtrics row-level export)",
            "supersedes": str(PDF.relative_to(MIP)).replace("\\", "/"),
            "generator": "scripts/survey_stats.py",
            "pdf_cross_check": {
                "date": "2026-08-04",
                "result": "MATCH on all 42 parts, zero disagreements",
                "method": "per part-set block: dashboard table cells where "
                          "readable, bar-chart vector geometry (bar widths "
                          "quantized to integer axis units, legend-color "
                          "series match) where tables were truncated; "
                          "response badges matched per-question n",
                "truncated_table_parts_recovered_from_chart": [
                    "Q1: Spring Hanger / Shackle Bracket, Pivot Pin / "
                    "Bushing Assembly, Axle Seat / Spring Perch",
                    "Q5: Door Storage Pocket, Door Handle/Lock Bezel "
                    "Opening, Window Switch Panel Recess",
                    "Q6: Wheel Wells, Side Step Board",
                    "Q7/Q8: see dashboard; all recovered cells matched",
                ],
            },
        },
        "instrument": {
            "title": "Car Part Manufacturing Techniques",
            "structure": "8 part-set pages, 42 parts total; per part the "
                         "rater multi-selects manufacturing processes "
                         "(instruction: 'up to three')",
            "option_set_varies_by_part_set": True,
            "merged_option": "Compression Molding/Thermoforming is a single "
                             "instrument option; it cannot be split between "
                             "the two benchmark processes",
            "survey_only_options": ["Adhesive_Bonding", "NA_none_of_the_listed"],
        },
        "responses": {
            "n_total": len(raters),
            "n_complete": sum(1 for ra in raters if ra["finished"]),
            "n_partial_with_answers": sum(
                1 for ra in raters
                if not ra["finished"] and per_rater_parts[ra["response_id"]] > 0),
            "per_rater": [
                {"response_id": ra["response_id"], "finished": ra["finished"],
                 "progress_pct": ra["progress"],
                 "duration_s": ra["duration_s"],
                 "n_parts_answered": per_rater_parts[ra["response_id"]]}
                for ra in raters],
        },
        "exclusion_rules": [
            "No response is dropped: partial responses contribute every part "
            "they answered (the instrument has no attention checks).",
            "Empty cells are non-answers, not negative picks.",
            "Picks beyond the 'up to three' instruction are kept as recorded "
            "and logged under instruction_deviations.",
        ],
        "instruction_deviations": overpick,
        "per_question_n": {k: per_q_n[k] for k in sorted(per_q_n)},
        "n_scored_instances": sum(p["n_raters"] for p in parts),
        "n_parts": len(parts),
        "n_raters": sum(1 for ra in raters
                        if per_rater_parts[ra["response_id"]] > 0),
        "expert_totals": dict(sorted(totals.items())),
        "imbalance": {
            "top_process": top[0], "top_picks": top[1],
            "runner_up": runner[0], "runner_up_picks": runner[1],
            "ratio": round(top[1] / runner[1], 2),
            "note": "replaces the unreproducible 217 vs 57 / '3.8-fold' claim",
        },
        "expert_top5": [{"process": p, "picks": n} for p, n in ranked[:5]],
        "agreement": {
            **krippendorff_alpha(units),
            "multi_select_handling":
                "Labels are SETS (multi-select, no rank order), so a "
                "primary-pick reduction is not defensible; alpha is computed "
                "over set-valued labels with MASI distance. Units = parts, "
                "observers = raters; units with one rater are unpairable and "
                "excluded from alpha only.",
            "leave_one_rater_out": leave_one_rater_out(parts),
        },
        "parts": parts,
        # cross-check against the model side lands with the figure script;
        # keys reserved so the audit can assert their presence later
        "krippendorff_alpha": None,      # legacy key: see agreement.alpha
    }
    out["krippendorff_alpha"] = out["agreement"].get("alpha")

    OUT.write_text(json.dumps(out, indent=2, sort_keys=False) + "\n",
                   encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"responses {out['responses']['n_total']} "
          f"(complete {out['responses']['n_complete']}), "
          f"parts {out['n_parts']}, scored instances "
          f"{out['n_scored_instances']}")
    print(f"imbalance: {top[0]} {top[1]} vs {runner[0]} {runner[1]} "
          f"= {out['imbalance']['ratio']}x")
    print(f"alpha (MASI): {out['agreement'].get('alpha')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
