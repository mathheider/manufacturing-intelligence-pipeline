"""Goal-directed active perception.

The planner picks the next feature to probe by expected feasibility-decision
information gain, not by semantic similarity. Heuristic: for each unobserved
feature, count how many *currently surviving* processes have a rule that
references it (and so could be eliminated by a confident answer). Probe the
highest-scoring feature next; stop when no unobserved feature could cut anything.

This is a cheap count-based proxy for true posterior-entropy information gain --
adequate for the prototype, and the place a dissertation-phase Bayesian planner
would slot in.
"""

from __future__ import annotations

from typing import Optional

from .rules import RuleSet


def expected_eliminations(survivors, rules: RuleSet, feature_name: str) -> int:
    """How many survivors reference ``feature_name`` in a forbidding rule."""
    return sum(1 for proc in survivors
               if feature_name in rules.features_referenced_by(proc))


def next_feature(graph, rules: RuleSet, exclude=None) -> Optional[str]:
    """Pick the next feature to probe, skipping ``exclude``.

    ``exclude`` is supplied by the agent: committed features plus features
    whose re-probe budget is spent (an unstable feature gets cfg.max_reprobe
    retries, then the agent abstains on it). The pre-budget behavior --
    re-picking a stubborn feature every step until max_steps -- multiplied
    real-backend API calls ~4x for no decision change.
    """
    if exclude is None:
        exclude = set(graph.features)
    candidates = [f for f in rules.features if f not in exclude]
    if not candidates:
        return None
    # Highest expected eliminations first; ties broken by name for determinism.
    scored = sorted(
        ((expected_eliminations(graph.survivors, rules, f), f) for f in candidates),
        key=lambda x: (-x[0], x[1]),
    )
    best_score, best_feature = scored[0]
    return best_feature if best_score > 0 else None


def question_for(feature_name: str, rules: RuleSet) -> str:
    fdef = rules.features[feature_name]
    if fdef.question:
        return fdef.question
    pretty = feature_name.replace("_", " ")
    if fdef.type == "numeric":
        unit = fdef.unit or "appropriate units"
        return f"Estimate the {pretty} in {unit}. Respond with a single number."
    if fdef.type == "boolean":
        return f"Is there a visible {pretty} in this component? Answer yes or no."
    if fdef.type == "integer":
        return f"How many {pretty}s are present? Answer with a non-negative integer."
    return f"Describe the {pretty} of this component."
