"""Golden-number regression: locks the deterministic benchmark output.

If any change perturbs the pipeline, seeding, or rules, these exact numbers
move and the test fails loudly. That is the contract behind "reproducible
results": the same code + same config + same seed always yields these values,
on any machine, under any PYTHONHASHSEED.

Config: 14 seed components, MockVLM, seeds=5, default noise (0.15 / 0.20),
n_samples=5. Point estimates only (bootstrap CIs depend on n_boot and are
checked for determinism separately).
"""

from mip.benchmark import run_benchmark
from mip.config import Config

# (precision, recall, f1, avg_false_elims, avg_probes)
# Re-locked 2026-07-01 after the Phase-1 densification (8 -> 11 features,
# 71 -> 92 clauses after adversarial counterexample review, new seed feature
# values). See docs/IMPROVEMENTS.md #17.
GOLDEN_ALL = {
    "Direct":  (0.557, 0.474, 0.498, 1.086, 1.000),
    "RuleSet": (0.234, 1.000, 0.360, 0.000, 0.000),
    "Greedy":  (0.173, 0.631, 0.252, 0.571, 11.000),
    "Hybrid":  (0.206, 0.855, 0.304, 0.186, 9.200),
}
GOLDEN_HARD = {
    "Direct":  (0.250, 0.104, 0.146, 1.875, 1.000),
    "RuleSet": (0.323, 1.000, 0.476, 0.000, 0.000),
    "Greedy":  (0.216, 0.554, 0.298, 0.800, 11.000),
    "Hybrid":  (0.284, 0.821, 0.399, 0.250, 9.275),
}
GOLDEN_GAPS = {
    "architecture_value_all": -0.194,
    "architecture_value_hard": 0.253,
    "perception_cost_all": 0.056,
    "perception_cost_hard": 0.077,
}


def _row(rows, m):
    r = rows[m]
    return (r["precision"], r["recall"], r["f1"],
            r["avg_false_elims"], r["avg_probes"])


def _summary():
    # module-level cache so we run the benchmark once for all assertions
    if not hasattr(_summary, "_cache"):
        _summary._cache = run_benchmark(
            Config(seeds=5), size=0, backend="mock", n_boot=200,
            out_path="outputs/_regression.json")
    return _summary._cache


def test_all_components_match_golden():
    rows = _summary()["all_components"]
    for m, expected in GOLDEN_ALL.items():
        assert _row(rows, m) == expected, f"{m} ALL drifted: {_row(rows, m)}"


def test_hard_subset_matches_golden():
    rows = _summary()["hard_subset"]
    for m, expected in GOLDEN_HARD.items():
        assert _row(rows, m) == expected, f"{m} HARD drifted: {_row(rows, m)}"


def test_diagnostic_gaps_match_golden():
    assert _summary()["diagnostic_gaps"] == GOLDEN_GAPS


def test_key_scientific_claims_hold():
    s = _summary()
    all_, hard = s["all_components"], s["hard_subset"]
    # 1. Corpus-frequency bias: Direct collapses on the HARD subset.
    assert hard["Direct"]["recall"] < 0.2
    # 2. Architecture value shows up exactly on HARD (Direct -> Hybrid).
    assert hard["Hybrid"]["f1"] > hard["Direct"]["f1"]
    # 3. Consistency sampling cuts false eliminations vs. Greedy.
    assert all_["Hybrid"]["avg_false_elims"] < all_["Greedy"]["avg_false_elims"]
    # 4. Hybrid recovers most of the perfect-perception (RuleSet) upper bound
    #    on HARD (>= 70% of oracle F1; the residual IS the perception cost).
    assert hard["Hybrid"]["f1"] / hard["RuleSet"]["f1"] >= 0.70
    # 5. Hybrid reaches its decision in fewer probes than Greedy.
    assert all_["Hybrid"]["avg_probes"] < all_["Greedy"]["avg_probes"]
