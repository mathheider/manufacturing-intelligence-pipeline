"""Rule-table loading and validation."""

import pytest

from mip.rules import Clause, Feature, RuleSet


def test_default_rules_load():
    rules = RuleSet.load()
    assert len(rules.processes) == 17
    assert len(rules.features) == 11
    assert rules.n_clauses == 92


def _minimal(features, rules):
    return RuleSet(processes=list(rules), features=features, rules=rules)


def test_rejects_unknown_feature():
    feats = {"wall": Feature("wall", "numeric")}
    with pytest.raises(ValueError, match="unknown feature"):
        _minimal(feats, {"P": [Clause("nope", "gt", 1.0, "", "")]})


def test_rejects_bad_operator():
    feats = {"wall": Feature("wall", "numeric")}
    with pytest.raises(ValueError, match="unknown operator"):
        _minimal(feats, {"P": [Clause("wall", "approx", 1.0, "", "")]})


def test_rejects_boolean_threshold_on_numeric():
    feats = {"wall": Feature("wall", "numeric")}
    with pytest.raises(ValueError, match="numeric"):
        _minimal(feats, {"P": [Clause("wall", "gt", True, "", "")]})


def test_rejects_ordered_op_on_boolean():
    feats = {"hollow": Feature("hollow", "boolean")}
    with pytest.raises(ValueError, match="eq/ne"):
        _minimal(feats, {"P": [Clause("hollow", "gt", True, "", "")]})


def test_every_clause_is_cited():
    rules = RuleSet.load()
    for proc in rules.processes:
        for clause in rules.clauses_for(proc):
            assert clause.cite, f"{proc}:{clause.feature} missing citation"
            assert clause.reason
