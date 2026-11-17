"""Rule engine: operator semantics, provenance, and conservatism."""

from mip.engine import check, eliminate
from mip.rules import RuleSet


def test_check_operators():
    assert check("gt", 9.0, 8.0) is True
    assert check("lt", 0.1, 0.3) is True
    assert check("gte", 8.0, 8.0) is True
    assert check("lte", 8.0, 8.0) is True
    assert check("eq", True, True) is True
    assert check("ne", True, False) is True


def test_check_none_never_fires():
    # Absence of evidence is not evidence of infeasibility.
    assert check("gt", None, 8.0) is False
    assert check("eq", None, True) is False


def test_check_bool_not_treated_as_ordered_number():
    # bool is an int subclass in Python; ordered ops must not silently compare it.
    assert check("gt", True, 0.5) is False


def test_hollow_eliminates_thermoforming_but_not_blow_molding():
    rules = RuleSet.load()
    feats = {"hollow": True}
    survivors, elims = eliminate(rules, feats)
    assert "Thermoforming" not in survivors
    assert "Blow_Molding" in survivors            # blow molding REQUIRES hollow
    assert "Injection_Molding" in survivors       # gas-assist IM allowed
    # elimination carries a citation
    assert elims["Thermoforming"].cite
    assert "hollow" in elims["Thermoforming"].as_text()


def test_solid_part_eliminates_blow_molding():
    rules = RuleSet.load()
    survivors, _ = eliminate(rules, {"hollow": False})
    assert "Blow_Molding" not in survivors
    assert "Filament_Winding" not in survivors     # also requires hollow


def test_eliminate_is_pure():
    rules = RuleSet.load()
    feats = {"hollow": True}
    before = dict(feats)
    eliminate(rules, feats)
    assert feats == before                          # inputs untouched


def test_empty_features_eliminate_nothing():
    rules = RuleSet.load()
    survivors, elims = eliminate(rules, {})
    assert set(survivors) == set(rules.processes)
    assert elims == {}
