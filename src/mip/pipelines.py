"""The four pipelines of the ablation, named to match the locked prelim story.

    Direct   (RAG + LLM)        one VLM call classifies the part directly.
                               Surfaces corpus-frequency bias. The baseline.
    RuleSet  (symbolic only)    rule engine on ORACLE (ground-truth) features.
                               Upper bound on rule fidelity given perfect
                               perception; uses no VLM.
    Greedy   (ablation)         perceive every feature once (noisy), commit
                               blindly, prune. No planning, no consistency.
                               Isolates the value of consistency sampling.
    Hybrid   (MIP agent)        goal-directed planning + consistency-calibrated
                               commitment + rule engine. The contribution.

The two diagnostic gaps the prelim reports:
    Direct  -> Hybrid  F1 gap  =  value added by the architecture
    RuleSet -> Hybrid  F1 gap  =  cost of doing perception instead of oracle
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Set

from .config import Config
from .engine import apply_to_graph, eliminate
from .planner import next_feature, question_for
from .rules import RuleSet
from .sampling import run_consistency
from .state import ConstraintGraph, FeatureObservation


@dataclass
class PipelineResult:
    name: str
    predicted: Set[str]
    n_probes: int                         # perception queries the method made
    graph: Optional[ConstraintGraph] = None
    extra: Dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Direct: RAG + LLM baseline
# --------------------------------------------------------------------------- #
def run_direct(vlm, image, rules: RuleSet) -> PipelineResult:
    pred = vlm.ask_process(image)
    return PipelineResult("Direct", {pred}, n_probes=1)


def run_direct_topk(vlm, image, rules: RuleSet, k: int) -> PipelineResult:
    """Set-size-matched Direct baseline (goal 3): the VLM emits its top-k
    ranked guesses instead of a singleton, so Direct's set size is comparable
    to Hybrid's survivor set. The ranked order is kept in ``extra`` for the
    shortlist metric."""
    ranked = list(vlm.ask_process_topk(image, k))
    return PipelineResult("Direct@k", set(ranked), n_probes=1,
                          extra={"ranked": ranked})


# --------------------------------------------------------------------------- #
# RuleSet: symbolic engine on oracle features (perfect perception upper bound)
# --------------------------------------------------------------------------- #
def run_ruleset(oracle_features: Mapping[str, object], rules: RuleSet) -> PipelineResult:
    survivors, elims = eliminate(rules, oracle_features)
    graph = ConstraintGraph(survivors=survivors)
    graph.eliminations = elims
    return PipelineResult("RuleSet", set(survivors), n_probes=0, graph=graph)


# --------------------------------------------------------------------------- #
# Greedy: perceive-all-once, no consistency, no planning
# --------------------------------------------------------------------------- #
def run_greedy(vlm, image, rules: RuleSet) -> PipelineResult:
    graph = ConstraintGraph(survivors=list(rules.processes))
    for fname, fdef in rules.features.items():
        resp = vlm.ask(image, question_for(fname, rules), fname, fdef.type)
        if resp.parsed is None:
            continue
        graph.commit(FeatureObservation(
            name=fname, value=resp.parsed, uncertainty=float("nan"),
            n_samples=1, committed=True, raw_samples=[resp.parsed]))
    apply_to_graph(graph, rules)
    return PipelineResult("Greedy", set(graph.survivors),
                          n_probes=len(rules.features), graph=graph)


# --------------------------------------------------------------------------- #
# Hybrid: the MIP agent (plan -> consistency-sample -> commit/skip -> prune)
# --------------------------------------------------------------------------- #
def run_hybrid(vlm, image, rules: RuleSet, cfg: Config,
               verbose: bool = False) -> PipelineResult:
    graph = ConstraintGraph(survivors=list(rules.processes))
    attempts: Dict[str, int] = {}      # feature -> probes made this episode

    for step in range(cfg.max_steps):
        if len(graph.survivors) <= 1:
            graph.log({"event": "terminate", "reason": "<=1 candidate", "step": step})
            break
        committed_names = {n for n, o in graph.features.items() if o.committed}
        spent = {f for f, n in attempts.items() if n > cfg.max_reprobe}
        fname = next_feature(graph, rules, exclude=committed_names | spent)
        if fname is None:
            graph.log({"event": "terminate", "reason": "no informative feature", "step": step})
            break
        attempts[fname] = attempts.get(fname, 0) + 1

        fdef = rules.features[fname]
        question = question_for(fname, rules)
        result = run_consistency(vlm, image, question, fname, fdef.type,
                                 n=cfg.n_samples, tol_rel=cfg.tol_rel,
                                 min_agree=cfg.min_agree)
        committed = bool(result.stable and result.value is not None)
        obs = FeatureObservation(fname, result.value, result.uncertainty,
                                 result.n_samples, committed, list(result.raw_samples))
        graph.features[fname] = obs
        graph.log({"event": "probe", "step": step, "feature": fname,
                   "samples": result.raw_samples, "value": result.value,
                   "uncertainty": result.uncertainty, "stable": result.stable})
        if verbose:
            print(f"[{step}] {fname}: {result.raw_samples} -> "
                  f"{'commit' if committed else 'SKIP'} {result.value}")
        if committed:
            apply_to_graph(graph, rules)

    graph.log({"event": "final", "survivors": graph.survivors})
    return PipelineResult("Hybrid", set(graph.survivors),
                          n_probes=len(graph.features), graph=graph,
                          extra={"committed": graph.n_committed()})


ALL_METHODS: List[str] = ["Direct", "RuleSet", "Greedy", "Hybrid"]
TOPK_METHOD = "Direct@k"
