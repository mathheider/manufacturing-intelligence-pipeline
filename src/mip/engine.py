"""The symbolic decision engine: eliminate infeasible processes from a set of
committed feature observations, recording *cited* provenance for every cut.

This is the "decision" half of the architecture. Perception (the VLM) never
touches it; it consumes only committed FeatureObservations. Keeping the engine
pure and side-effect-free (it returns a new survivor set + eliminations rather
than mutating global state) is what makes the whole pipeline deterministic and
unit-testable.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Tuple

from .rules import Clause, RuleSet
from .state import Elimination

_OPS = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "gt": lambda a, b: a > b,
    "lt": lambda a, b: a < b,
    "gte": lambda a, b: a >= b,
    "lte": lambda a, b: a <= b,
}


def check(op: str, observed: Any, threshold: Any) -> bool:
    """Return True iff the observed value satisfies the forbidding predicate.

    A missing (None) observation never fires a rule -- absence of evidence is
    not evidence of infeasibility. This conservatism is deliberate: the pruner
    only cuts on positive evidence.
    """
    if observed is None:
        return False
    # Guard against bool/int cross-comparison surprises for ordered ops.
    if op in ("gt", "lt", "gte", "lte") and isinstance(observed, bool):
        return False
    return _OPS[op](observed, threshold)


def _fires(clause: Clause, features: Mapping[str, Any]) -> bool:
    obs = features.get(clause.feature)
    return check(clause.op, obs, clause.value)


def eliminate(rules: RuleSet, features: Mapping[str, Any],
              candidates: Iterable[str] | None = None
              ) -> Tuple[List[str], Dict[str, Elimination]]:
    """Given committed feature *values* (name -> value), return
    ``(survivors, eliminations)`` where eliminations maps each cut process to
    the first Clause that eliminated it, with cited provenance.

    Pure function: does not mutate its inputs.
    """
    candidates = list(candidates) if candidates is not None else list(rules.processes)
    survivors: List[str] = []
    eliminations: Dict[str, Elimination] = {}
    for proc in candidates:
        cut: Elimination | None = None
        for clause in rules.clauses_for(proc):
            if _fires(clause, features):
                cut = Elimination(
                    process=proc, feature=clause.feature, op=clause.op,
                    value=clause.value, observed=features.get(clause.feature),
                    cite=clause.cite, reason=clause.reason)
                break
        if cut is None:
            survivors.append(proc)
        else:
            eliminations[proc] = cut
    return survivors, eliminations


def shortlist(survivors: Iterable[str], rules: RuleSet,
              observed: Iterable[str], k: int) -> List[str]:
    """Rank ``survivors`` by symbolic constraint evidence and return the top-k.

    Score(process) = number of the process's cited clauses whose feature was
    actually observed (each such clause was evaluated and did NOT fire -- the
    process survived it). More checked-and-passed constraints means more
    positive symbolic evidence of feasibility. The ranking uses only the rule
    table and the set of observed features; no learned component and no VLM --
    the categorical call never goes back to the net. Ties break alphabetically
    for determinism.
    """
    obs = set(observed)
    scored = sorted(
        ((sum(1 for c in rules.clauses_for(p) if c.feature in obs), p)
         for p in survivors),
        key=lambda t: (-t[0], t[1]))
    return [p for _, p in scored[:max(0, k)]]


def apply_to_graph(graph, rules: RuleSet) -> None:
    """Prune ``graph.survivors`` in place using its committed features, writing
    Elimination provenance into ``graph.eliminations`` and the trace."""
    values = {name: obs.value for name, obs in graph.committed_features().items()}
    survivors, elims = eliminate(rules, values, graph.survivors)
    for proc, elim in elims.items():
        if proc not in graph.eliminations:
            graph.eliminations[proc] = elim
            graph.log({"event": "prune", "process": proc,
                       "provenance": elim.to_dict()})
    graph.survivors = survivors
