"""Tests for the Phase-1/Phase-2 additions (CLAUDE_CODE_GOALS.md goals 1-4, 8):
densified rule table, elimination-native metrics, the set-size-matched
Direct@k mode, the symbolic shortlist ranking, and the real-image loader.
"""

import json
from pathlib import Path

from mip.benchmark import run_benchmark
from mip.config import Config
from mip.dataset import load_real, load_seed
from mip.engine import shortlist
from mip.metrics import MethodAggregate, score, summarize
from mip.planner import question_for
from mip.rules import RuleSet
from mip.vlm import MockVLM
from mip.vlm.real import parse_topk


# ---------------------------------------------------------------------- rules

def test_rule_table_densified():
    """71 -> 92: 30 clauses were drafted, 9 were refuted with real-world
    counterexamples in adversarial review and removed (correctness over the
    round '100 rules' tagline; artifacts must state 92)."""
    rules = RuleSet.load()
    assert len(rules.processes) == 17
    assert len(rules.features) == 11
    assert rules.n_clauses == 92


def test_every_clause_carries_cite_and_reason():
    rules = RuleSet.load()
    for proc in rules.processes:
        for c in rules.clauses_for(proc):
            assert c.cite.strip(), f"{proc}: clause on {c.feature} missing cite"
            assert c.reason.strip(), f"{proc}: clause on {c.feature} missing reason"


def test_verify_flagged_clauses_only_use_expected_format():
    """Proposed citations must be machine-findable for citation_status.py."""
    rules = RuleSet.load()
    n_verify = sum(1 for p in rules.processes for c in rules.clauses_for(p)
                   if c.cite.startswith("[VERIFY"))
    assert n_verify > 0                      # Phase-1 proposals exist...
    for proc in rules.processes:
        for c in rules.clauses_for(proc):
            if "[VERIFY" in c.cite:
                assert c.cite.startswith("[VERIFY:"), (proc, c.feature, c.cite)


def test_new_features_have_explicit_vlm_questions():
    rules = RuleSet.load()
    q = question_for("production_volume", rules)
    assert "0 for a one-off" in q
    assert "yes or no" in question_for("continuous_fiber_required", rules)
    assert "millimetres" in question_for("max_dimension_mm", rules)


# ------------------------------------------------------------------ shortlist

def test_shortlist_is_deterministic_topk_by_checked_constraints():
    rules = RuleSet.load()
    observed = set(rules.features)           # oracle: everything observed
    top = shortlist(rules.processes, rules, observed, 3)
    assert len(top) == 3
    assert top == shortlist(rules.processes, rules, observed, 3)
    # with nothing observed, ranking degrades to alphabetical
    assert shortlist(["B_proc", "A_proc"], rules, set(), 2) == ["A_proc", "B_proc"]


# ------------------------------------------------------------------- Direct@k

def test_mock_topk_is_deterministic_distinct_and_biased():
    a = MockVLM(seed=7).ask_process_topk({}, 5)
    b = MockVLM(seed=7).ask_process_topk({}, 5)
    assert a == b and len(a) == 5 and len(set(a)) == 5
    # corpus-frequency prior: IM leads the ranking almost always
    assert "Injection_Molding" in a[:2]
    assert len(MockVLM(seed=7).ask_process_topk({}, 17)) == 17


def test_parse_topk_matches_and_fills_to_k():
    got = parse_topk("1. Injection Molding\n2. thermoforming", 4)
    assert got[0] == "Injection_Molding" and got[1] == "Thermoforming"
    assert len(got) == 4 and len(set(got)) == 4
    assert len(parse_topk("", 3)) == 3       # unparseable -> deterministic fill


# -------------------------------------------------------------------- metrics

def test_summarize_carries_elimination_native_columns():
    agg = MethodAggregate()
    s = score({"A", "B"}, {"A"})
    agg.add(s, probes=3, contained=True, set_size=2, shortlist_f1=s.f1)
    row = summarize(agg, ("t",), n_boot=10, n_processes=17)
    assert row["containment"] == 1.0
    assert row["avg_set_size"] == 2.0
    assert row["survivor_reduction"] == 15.0
    assert "shortlist_f1" in row


def test_benchmark_reports_topk_row_and_ruleset_containment_invariant():
    summary = run_benchmark(Config(seeds=2), size=0, backend="mock",
                            n_boot=50, direct_topk=5,
                            out_path="outputs/_test_phase1.json")
    rows = summary["all_components"]
    assert "Direct@k" in rows
    assert rows["Direct@k"]["avg_set_size"] == 5.0
    # dataset invariant: oracle rules never eliminate an expert process
    assert rows["RuleSet"]["containment"] == 1.0
    assert rows["RuleSet"]["recall"] == 1.0
    for m in rows:
        assert "survivor_reduction" in rows[m]


# ----------------------------------------------------------------- seed data

def test_seed_components_carry_new_features():
    for inst in load_seed():
        assert "max_dimension_mm" in inst.features, inst.id
        assert inst.features["production_volume"] in (0, 1, 2), inst.id
        assert isinstance(inst.features["continuous_fiber_required"], bool), inst.id


# ------------------------------------------------------------------ load_real

def test_load_real_skips_missing_files(tmp_path):
    m = tmp_path / "m.json"
    m.write_text(json.dumps({"images": [{
        "path": "images_pool/nope/nope_000.jpg", "category": "dashboard",
        "expert_processes": ["Injection_Molding"], "source": "x", "url": "u",
        "license": "CC0", "attribution": "a", "width": 900, "height": 600,
        "ahash": "0" * 64}]}), encoding="utf-8")
    assert load_real(manifest=m, accepted_only=False) == []


# ------------------------------------------------------------- rules override

def test_bench_rules_path_override(tmp_path):
    """--rules points the benchmark at an alternate rules JSON (robustness
    ablations, e.g. verified-clauses-only) and the summary records which
    rule set produced it. The subset is carved structurally (drop each
    process's last clause), never by citation markers or a hardcoded clause
    census, so the test survives citation edits to the live rules.json."""
    raw = json.loads((Path(__file__).parents[1] / "data" / "rules.json")
                     .read_text(encoding="utf-8"))
    total = sum(len(b["forbidden_if"]) for b in raw["rules"].values())
    dropped = 0
    for body in raw["rules"].values():
        if body["forbidden_if"]:
            body["forbidden_if"] = body["forbidden_if"][:-1]
            dropped += 1
    assert 0 < dropped < total
    alt = tmp_path / "rules_subset.json"
    alt.write_text(json.dumps(raw), encoding="utf-8")

    cfg = Config(seeds=2)
    summary = run_benchmark(cfg, n_boot=10, direct_topk=8,
                            out_path=tmp_path / "s.json", rules_path=alt)
    assert summary["rules_path"] == str(alt)
    assert summary["n_clauses"] == total - dropped
    default = run_benchmark(Config(seeds=2), n_boot=10, direct_topk=8,
                            out_path=tmp_path / "d.json")
    assert default["rules_path"] == "data/rules.json"
    assert default["n_clauses"] == total
