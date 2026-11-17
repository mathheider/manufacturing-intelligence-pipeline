"""Pipeline behaviour and the RuleSet upper-bound property."""

from mip.config import Config
from mip.dataset import load_seed
from mip.pipelines import run_direct, run_greedy, run_hybrid, run_ruleset
from mip.rules import RuleSet
from mip.vlm import MockVLM


def _cf_cylinder():
    return {i.id: i for i in load_seed()}["cf_cylinder"]


def test_ruleset_oracle_never_drops_expert_process():
    rules = RuleSet.load()
    for inst in load_seed():
        res = run_ruleset(inst.features, rules)
        assert inst.expert_processes.issubset(res.predicted)
        assert res.n_probes == 0


def test_direct_returns_single_prediction():
    rules = RuleSet.load()
    res = run_direct(MockVLM(seed=0), {"wall_thickness_mm": 3.0}, rules)
    assert len(res.predicted) == 1
    assert res.n_probes == 1


def test_hybrid_produces_cited_provenance():
    rules = RuleSet.load()
    inst = _cf_cylinder()
    res = run_hybrid(MockVLM(seed=5), inst.features, rules, Config())
    assert res.graph is not None
    # cf_cylinder is hollow+axisymmetric: many matched-die processes must be cut
    assert res.graph.eliminations
    for elim in res.graph.eliminations.values():
        assert elim.cite                       # every cut is cited
        assert elim.feature in rules.features


def test_hybrid_probes_fewer_than_greedy():
    rules = RuleSet.load()
    inst = _cf_cylinder()
    hy = run_hybrid(MockVLM(seed=5), inst.features, rules, Config())
    gr = run_greedy(MockVLM(seed=5), inst.features, rules)
    assert hy.n_probes <= gr.n_probes


def test_hybrid_graph_is_json_serialisable():
    import json
    rules = RuleSet.load()
    inst = _cf_cylinder()
    res = run_hybrid(MockVLM(seed=5), inst.features, rules, Config())
    json.dumps(res.graph.to_dict())            # must not raise
