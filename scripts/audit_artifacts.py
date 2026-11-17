# -*- coding: utf-8 -*-
"""Cross-artifact number audit (goal 18, CLAUDE_CODE_GOALS.md).

Proves that every headline number in the deck and report appears in an
outputs/*.json produced by a logged command, and that rule/feature/dataset
counts are identical across README, deck, report, and PROJECT_BRIEF.

    python scripts/audit_artifacts.py          # exit 1 on any mismatch

Requires: python-pptx, python-docx (audit-only dependencies; the benchmark
itself stays zero-dep).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

MIP = Path(__file__).resolve().parent.parent
ROOT = MIP.parent
DECK = ROOT / "Prelim_Deck_DEFENSE.pptx"
REPORT = ROOT / "Prelim_Report_Heider_DRAFT.docx"

FAIL = []
WARN = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok    {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}  {detail}")


def deck_text() -> str:
    from pptx import Presentation
    p = Presentation(DECK)
    out = []
    for s in p.slides:
        for sh in s.shapes:
            if sh.has_text_frame:
                out.append(sh.text_frame.text)
            if getattr(sh, "has_table", False) and sh.has_table:
                for r in sh.table.rows:
                    out.extend(c.text for c in r.cells)
        if s.has_notes_slide:
            out.append(s.notes_slide.notes_text_frame.text)
    return "\n".join(out)


def report_text() -> str:
    import docx
    d = docx.Document(REPORT)
    out = [p.text for p in d.paragraphs]
    for t in d.tables:
        for r in t.rows:
            out.extend(c.text for c in r.cells)
    return "\n".join(out)


def main() -> int:
    rules = json.loads((MIP / "data" / "rules.json").read_text(encoding="utf-8"))
    n_clauses = sum(len(b["forbidden_if"]) for b in rules["rules"].values())
    n_verify = sum(1 for b in rules["rules"].values()
                   for c in b["forbidden_if"] if c["cite"].startswith("[VERIFY"))
    n_feat = len(rules["features"])
    n_proc = len(rules["processes"])
    comps = json.loads((MIP / "data" / "components.json").read_text(encoding="utf-8"))
    b1000 = json.loads((MIP / "outputs" / "benchmark_1000.json").read_text(encoding="utf-8"))
    bsum = json.loads((MIP / "outputs" / "benchmark_summary.json").read_text(encoding="utf-8"))

    print(f"artifact ground truth: {n_clauses} clauses ({n_verify} [VERIFY]), "
          f"{n_feat} features, {n_proc} processes, {len(comps)} seed components")

    print("\n== data invariants ==")
    check("rule count is 92", n_clauses == 92, f"got {n_clauses}")
    check("features 11 / processes 17 / seeds 14",
          (n_feat, n_proc, len(comps)) == (11, 17, 14))

    dk = deck_text()
    rp = report_text()

    print("\n== stale-claim scan (deck) ==")
    for bad in ("100 cited", "100 rule", "one hundred cited", "100-rule",
                "0.77", ".77 ", "0.58", "0.69", "eight features",
                "precise but abstains", "Locked run"):
        check(f"deck free of {bad!r}", bad not in dk)
    print("\n== stale-claim scan (report) ==")
    rp_low = rp.lower()
    for bad in ("100 cited", "100 rule", "100 forbidden", "100 source-cited",
                "0.77", "0.58", "0.69", "three-way ablation",
                "precise but sparse", "217 automotive component instances",
                "three end-to-end pipeline implementations"):
        check(f"report free of {bad!r}", bad.lower() not in rp_low)

    print("\n== headline numbers traceable to outputs/benchmark_1000.json ==")
    hb = b1000["hard_subset"]
    ab = b1000["all_components"]

    def has(text, v):
        s = f"{v:.3f}"
        return s in text or s.lstrip("0") in text

    # every value below is READ from the JSON; the check is that the deck and
    # report carry exactly the JSON's value (not a stale or rounded variant)
    for name, v, where in [
        ("Direct HARD F1", hb["Direct"]["f1"], "both"),
        ("Direct HARD recall", hb["Direct"]["recall"], "both"),
        ("Hybrid HARD F1", hb["Hybrid"]["f1"], "both"),
        ("RuleSet HARD F1", hb["RuleSet"]["f1"], "both"),
        ("architecture gap HARD", b1000["diagnostic_gaps"]["architecture_value_hard"], "both"),
        ("Direct ALL F1", ab["Direct"]["f1"], "both"),
        ("Direct@k ALL F1", ab["Direct@k"]["f1"], "both"),
        ("Direct@k HARD F1", hb["Direct@k"]["f1"], "both"),
        ("Hybrid ALL F1", ab["Hybrid"]["f1"], "both"),
        ("Hybrid HARD CI low", hb["Hybrid"]["f1_ci95"][0], "both"),
        ("Hybrid HARD CI high", hb["Hybrid"]["f1_ci95"][1], "both"),
    ]:
        ok = has(dk, v) and has(rp, v) if where == "both" else has(dk, v)
        check(f"{name} = {v:.3f} present in deck+report", ok)
    check("topk k=8 named in both",
          b1000["direct_topk"] == 8 and "Direct@8" in dk and "Direct@8" in rp)
    check("falseElim mechanism numbers present in deck",
          f"{ab['Greedy']['avg_false_elims']:.2f}" in dk
          and f"{ab['Hybrid']['avg_false_elims']:.2f}" in dk)

    real_p = MIP / "outputs" / "benchmark_real.json"
    if real_p.exists():
        print("\n== real-image numbers traceable to outputs/benchmark_real.json ==")
        br = json.loads(real_p.read_text(encoding="utf-8"))
        rab, rhb = br["all_components"], br["hard_subset"]
        for name, v in [
            ("real Direct ALL F1", rab["Direct"]["f1"]),
            ("real Direct HARD F1", rhb["Direct"]["f1"]),
            ("real Direct@8 HARD F1", rhb["Direct@k"]["f1"]),
            ("real Hybrid HARD F1", rhb["Hybrid"]["f1"]),
        ]:
            check(f"{name} = {v:.3f} present in deck+report", has(dk, v) and has(rp, v))
        check("real containment .835 present",
              abs(rhb["Direct@k"]["containment"] - 0.835) < 1e-9
              and (".835" in rp or "0.835" in rp))
        check("real dataset size 150 stated",
              br["dataset_size"] == 150 and "150" in dk and "150" in rp)

    interim_p = MIP / "outputs" / "benchmark_real_v2_interim.json"
    if "interim" in dk.lower() or interim_p.exists():
        print("\n== bench-v2 interim readout traceability ==")
        if not interim_p.exists():
            check("deck INTERIM numbers trace to outputs/benchmark_real_v2_interim.json",
                  False, "deck cites an interim readout but the artifact is missing; "
                  "sync it from the run machine or regenerate it with "
                  "scripts/score_v2_interim.py, then rerun "
                  "scripts/make_v2_interim_arc_chart.py")
        else:
            ivm = json.loads(interim_p.read_text(encoding="utf-8"))
            check("interim artifact is marked INTERIM with the disclosure note",
                  bool(ivm.get("INTERIM")) and "deviation" in str(ivm.get("note", "")).lower())
            check("interim artifact is bench-v2 on the closed 635 pool",
                  ivm.get("prompt_version") == "bench-v2"
                  and ivm.get("pool_size") == 635)
            for name, v in [
                ("interim Hybrid HARD F1", ivm["hard_subset"]["Hybrid"]["f1"]),
                ("interim Hybrid ALL F1", ivm["all_components"]["Hybrid"]["f1"]),
            ]:
                check(f"{name} = {v:.3f} present in deck", has(dk, v))
            if not has(rp, ivm["hard_subset"]["Hybrid"]["f1"]):
                print("  note  report 8.6 does not carry the interim number yet "
                      "(CC-7 scope)")

    print("\n== canonical README block matches benchmark_summary.json ==")
    readme = (MIP / "README.md").read_text(encoding="utf-8")
    for m, r in bsum["all_components"].items():
        row_ok = f"{r['f1']:.3f}" in readme
        check(f"README carries {m} ALL F1 {r['f1']:.3f}", row_ok)

    print("\n== manifest licensing ==")
    man = json.loads((MIP / "images_pool" / "images_manifest.json").read_text(encoding="utf-8"))
    ok_lic = re.compile(r"(CC0|CC.?BY(.?SA)?|Public domain|PD)", re.IGNORECASE)
    bad = [e["path"] for e in man["images"]
           if not (ok_lic.search(e.get("license", "")) and e.get("attribution"))]
    check(f"every pool image has license+attribution ({len(man['images'])} entries)",
          not bad, f"{len(bad)} missing")

    print("\n== survey status ==")
    survey_p = MIP / "outputs" / "survey_stats.json"
    if survey_p.exists():
        # CC-1 landed: survey-derived claims must trace to survey_stats.json
        sv = json.loads(survey_p.read_text(encoding="utf-8"))
        check("survey_stats.json has n/raters/alpha keys",
              all(k in sv for k in ("n_scored_instances", "n_raters",
                                    "krippendorff_alpha")))
        check("deck TO CONFIRM survey flag removed", "TO CONFIRM" not in dk)
        check("report TO CONFIRM survey flag removed", "TO CONFIRM" not in rp)
        # CC-1 DO#4 (wired 2026-08-04): survey-derived numbers in deck and
        # report trace to the committed stats JSONs; every value is READ from
        # the JSON, never retyped here.
        obs = json.loads((MIP / "outputs" / "observation_stats.json").read_text(encoding="utf-8"))
        dps = json.loads((MIP / "outputs" / "decision_paired_stats.json").read_text(encoding="utf-8"))
        im = sv["imbalance"]
        check("survey picks 103 vs 57 in deck+report",
              im["top_picks"] == 103 and im["runner_up_picks"] == 57
              and "103" in dk and "103" in rp and "57" in dk and "57" in rp)
        check("imbalance ratio 1.81 present; 3.8-fold retired",
              f"{im['ratio']:.2f}" == "1.81" and "1.81" in dk and "1.81" in rp
              and "three-point-eight" not in dk and "3.8-fold" not in dk
              and "3.8 times as often" not in rp)
        check("alpha 0.22 stated in deck notes + report",
              round(sv["krippendorff_alpha"], 2) == 0.22
              and "0.22" in dk and "0.22" in rp)
        check("protocol 13 respondents / 42 parts / 163 ratings stated",
              sv["responses"]["n_total"] == 13 and sv["n_parts"] == 42
              and sv["n_scored_instances"] == 163
              and all(t in dk for t in ("13 respondents", "42 parts", "163"))
              and all(t in rp for t in ("13 respondents", "42 parts", "163")))
        check("unreproducible 217 retired from deck",
              "217" not in dk and "two hundred seventeen" not in dk)
        check("model 71.3% vs expert 24.1% shares traced",
              abs(obs["model"]["top_share"] - 0.713) < 1e-9
              and abs(obs["experts"]["top_share"] - 0.241) < 1e-9
              and "71" in dk and "24" in dk and "71.3" in rp and "24.1" in rp)
        check("top-5 overlap = 3 (replaces 'only one')",
              obs["top5_overlap"]["count"] == 3 and "three processes" in dk
              and "three of five" in rp and "on only one process" not in dk
              and "on only one of five" not in rp)
        pcls = dps["paired_cells"]
        check("paired decider test 129 discordant (128 vs 1) present",
              pcls["direct_only"] == 128 and pcls["hybrid_only"] == 1
              and "129" in dk and "128" in dk and "129" in rp and "128" in rp)
        check("binary containment 88.7 vs 4.0 present",
              dps["containment"]["direct_at_8"] == 133
              and dps["containment"]["hybrid"] == 6
              and "88.7" in dk and "4.0" in dk and "88.7" in rp and "4.0" in rp)
        check("McNemar exact p 3.8e-37 stated",
              dps["mcnemar_exact_p"] < 1e-36
              and "3.8e-37" in dk and "3.8e-37" in rp)
        check("retired retrieval claim absent",
              "Retrieval Does Not Move the Decision" not in dk
              and "unchanged from the no-retrieval baseline" not in dk
              and "It does not. When handbook passages" not in rp)
    else:
        # goal 14 unresolved -> deck and report must still flag the survey
        flag_dk = ("TO CONFIRM" in dk) or ("reconciliation is in progress" in dk)
        check("deck flags unresolved survey "
              "(TO CONFIRM / 'reconciliation is in progress')", flag_dk)
        check("report carries TO CONFIRM survey flag", "TO CONFIRM" in rp)

    if FAIL:
        print(f"\nAUDIT FAILED: {len(FAIL)} problem(s)")
        return 1
    print("\nAUDIT PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
