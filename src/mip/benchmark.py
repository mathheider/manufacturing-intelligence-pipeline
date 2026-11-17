"""Deterministic ablation runner.

For every (instance, trial) we derive ONE seed and build a fresh MockVLM with it
for each pipeline. Because MockVLM's noise is a pure function of
(seed, feature, sample_index), every pipeline perceives the *identical* noisy
feature reads -- the only thing that varies between Direct / Greedy / Hybrid is
the decision logic. RuleSet uses oracle features and no VLM.

Datasets:
* ``dataset="seed"``/``"synthetic"`` -- the offline deterministic path
  (``--size`` selects 14 seeds or a synthetic expansion).
* ``dataset="real"`` -- the CC-licensed photo pool (category-level expert
  labels). No feature oracle exists for photos, so the RuleSet upper bound is
  skipped. Real backends are wrapped in a disk cache (``vlm/cache.py``) so the
  run is resumable and every raw response is archived.

Outputs a summary dict (config echoed in, bootstrap CIs, elimination-native
metrics, the diagnostic gaps) and, optionally, per-instance Hybrid provenance
graphs.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional

from ._seed import derive_seed
from .config import Config
from .dataset import Instance, load_dataset, load_real
from .engine import shortlist
from .metrics import MethodAggregate, score, summarize
from .pipelines import (ALL_METHODS, TOPK_METHOD, run_direct, run_direct_topk,
                        run_greedy, run_hybrid, run_ruleset)
from .rules import RuleSet
from .vlm import MockVLM, get_backend

_OUT = Path(__file__).resolve().parents[2] / "outputs"


def _observed_features(method: str, res, instance: Instance) -> set:
    """The features each method actually had in hand -- the evidence base for
    the symbolic shortlist ranking."""
    if method == "RuleSet":
        return set(instance.features)
    if res.graph is not None:
        return set(res.graph.committed_features())
    return set()


def _run_all_pipelines(instance: Instance, rules: RuleSet, cfg: Config,
                       backend: str, real_vlm=None, methods=None,
                       direct_topk: int = 0) -> Dict:
    """Return {method: PipelineResult} for one instance/trial."""
    methods = methods or ALL_METHODS
    out: Dict = {}
    if backend == "mock":
        seed = derive_seed(cfg.seed, instance.id)
        image = instance.features                       # MockVLM reads truth here
        make = lambda: MockVLM(noise=cfg.noise, seed=seed)
        if "Direct" in methods:
            out["Direct"] = run_direct(make(), image, rules)
        if direct_topk > 0:
            out[TOPK_METHOD] = run_direct_topk(make(), image, rules, direct_topk)
        if "Greedy" in methods:
            out["Greedy"] = run_greedy(make(), image, rules)
        if "Hybrid" in methods:
            out["Hybrid"] = run_hybrid(make(), image, rules, cfg)
    else:
        image = str((_OUT.parent / instance.image)) if instance.image else None
        if "Direct" in methods:
            real_vlm.prefix = "direct:"
            out["Direct"] = run_direct(real_vlm, image, rules)
        if direct_topk > 0:
            real_vlm.prefix = "topk:"
            out[TOPK_METHOD] = run_direct_topk(real_vlm, image, rules, direct_topk)
        if "Greedy" in methods:
            real_vlm.prefix = "greedy:"
            out["Greedy"] = run_greedy(real_vlm, image, rules)
        if "Hybrid" in methods:
            real_vlm.prefix = "hybrid:"
            out["Hybrid"] = run_hybrid(real_vlm, image, rules, cfg)
    if "RuleSet" in methods:
        out["RuleSet"] = run_ruleset(instance.features, rules)  # oracle, no VLM
    return out


def _score_result(method: str, res, instance: Instance, rules: RuleSet,
                  shortlist_k: int):
    s = score(res.predicted, instance.expert_processes)
    contained = instance.expert_processes.issubset(res.predicted)
    if res.extra.get("ranked"):
        short = list(res.extra["ranked"])[:shortlist_k]
    elif res.graph is not None or method == "RuleSet":
        short = shortlist(res.predicted, rules,
                          _observed_features(method, res, instance), shortlist_k)
    else:
        short = sorted(res.predicted)[:shortlist_k]
    s_short = score(set(short), instance.expert_processes)
    return s, contained, len(res.predicted), s_short.f1


def run_benchmark(cfg: Optional[Config] = None, size: int = 0,
                  backend: str = "mock", n_boot: int = 1000,
                  save_graphs: bool = False,
                  out_path: Optional[Path] = None,
                  dataset: str = "seed",
                  limit: int = 0,
                  direct_topk: int = 0,
                  shortlist_k: int = 3,
                  workers: int = 1,
                  model: Optional[str] = None,
                  stratified: bool = False,
                  rules_path: Optional[Path] = None) -> Dict:
    cfg = cfg or Config()
    rules = RuleSet.load(rules_path)

    if dataset == "real":
        instances = load_real(stratified=stratified)
        if limit:
            instances = instances[:limit]
        if backend == "mock":
            raise ValueError("the real dataset needs --backend gemini|claude "
                             "(photos carry no ground-truth features for MockVLM)")
        methods = [m for m in ALL_METHODS if m != "RuleSet"]
        trials = [0]
    else:
        instances = load_dataset(size=size, seed=cfg.seed)
        methods = list(ALL_METHODS)
        trials = list(range(cfg.seeds)) if backend == "mock" else [0]

    real_inner = None
    real_cache_dir = None
    cache_stats = {"hits": 0, "misses": 0}
    _stats_lock = threading.Lock()
    if backend != "mock":
        # one cache dir PER MODEL so responses from different models never mix
        kw = {"model": model} if model else {}
        cache_name = "bench" if not model else f"bench-{model}"
        real_inner = get_backend(backend, **kw)
        real_cache_dir = _OUT / "vlm_cache" / cache_name

    if direct_topk < 0:                       # auto: match Hybrid's set size
        if backend != "mock":
            raise ValueError("--direct-topk auto needs the mock backend; give "
                             "an explicit k for real runs")
        direct_topk = _auto_topk(cfg, instances, rules)

    method_rows = methods + ([TOPK_METHOD] if direct_topk > 0 else [])
    agg = {m: MethodAggregate() for m in method_rows}
    agg_hard = {m: MethodAggregate() for m in method_rows}
    graphs: List[Dict] = []

    for trial in trials:
        trial_cfg = cfg if backend != "mock" else _with_trial(cfg, trial)
        todo = []
        for inst in instances:
            if backend != "mock":
                img = _OUT.parent / (inst.image or "")
                if not inst.image or not img.exists():
                    continue
            todo.append(inst)

        def _one(inst):
            # THREAD-SAFETY (2026-08-05): CachedVLM carries per-call mutable
            # state (channel prefix, per-(sha, channel) sample index). One
            # wrapper shared across worker threads races that state and
            # mis-keys the archived responses, which silently breaks
            # replay/resume (found via the interim replay attempt; first
            # window quarantined to vlm_cache/bench_raced_20260805). A fresh
            # wrapper PER INSTANCE shares the inner client but keeps the
            # archive deterministic: each instance is a distinct image sha.
            vlm = None
            if backend != "mock":
                from .vlm.cache import CachedVLM
                vlm = CachedVLM(real_inner, real_cache_dir)
            res = _run_all_pipelines(inst, rules, trial_cfg, backend, vlm,
                                     methods, direct_topk)
            if vlm is not None:
                with _stats_lock:
                    cache_stats["hits"] += vlm.hits
                    cache_stats["misses"] += vlm.misses
            return res

        if backend != "mock" and workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers) as pool:
                all_results = list(pool.map(_one, todo))
        else:
            all_results = [_one(i) for i in todo]

        for inst, results in zip(todo, all_results):
            hard = inst.expert_processes != {"Injection_Molding"}
            for m, res in results.items():
                s, contained, set_size, short_f1 = _score_result(
                    m, res, inst, rules, shortlist_k)
                agg[m].add(s, res.n_probes, contained, set_size, short_f1)
                if hard:
                    agg_hard[m].add(s, res.n_probes, contained, set_size, short_f1)
            if save_graphs and trial == trials[0] and results.get("Hybrid") and \
                    results["Hybrid"].graph:
                graphs.append({"instance": inst.id,
                               "expert": sorted(inst.expert_processes),
                               "graph": results["Hybrid"].graph.to_dict()})

    n_proc = len(rules.processes)
    all_rows = {m: summarize(agg[m], ("all", m), n_boot, n_proc) for m in method_rows}
    hard_rows = {m: summarize(agg_hard[m], ("hard", m), n_boot, n_proc)
                 for m in method_rows}

    summary = {
        "backend": backend,
        "dataset": dataset,
        "dataset_size": len(instances),
        "trials": len(trials),
        "direct_topk": direct_topk,
        "shortlist_k": shortlist_k,
        "rules_path": str(rules_path) if rules_path else "data/rules.json",
        "n_clauses": rules.n_clauses,
        "config": cfg.to_dict(),
        "all_components": all_rows,
        "hard_subset": hard_rows,
        "diagnostic_gaps": _gaps(all_rows, hard_rows),
    }
    if backend != "mock":
        summary["vlm_cache"] = dict(cache_stats)
    out_path = Path(out_path) if out_path else (_OUT / "benchmark_summary.json")
    _OUT.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if save_graphs:
        (_OUT / "hybrid_graphs.json").write_text(
            json.dumps(graphs, indent=2), encoding="utf-8")
    summary["_out_path"] = str(out_path)
    return summary


def _auto_topk(cfg: Config, instances: List[Instance], rules: RuleSet) -> int:
    """k for the set-size-matched Direct baseline: Hybrid's mean survivor-set
    size over a single deterministic trial, rounded to nearest int."""
    trial_cfg = _with_trial(cfg, 0)
    sizes = []
    for inst in instances:
        seed = derive_seed(trial_cfg.seed, inst.id)
        vlm = MockVLM(noise=trial_cfg.noise, seed=seed)
        res = run_hybrid(vlm, inst.features, rules, trial_cfg)
        sizes.append(len(res.predicted))
    return max(1, round(sum(sizes) / len(sizes))) if sizes else 1


def _with_trial(cfg: Config, trial: int) -> Config:
    """A per-trial config whose master seed folds in the trial index."""
    c = Config(seed=derive_seed(cfg.seed, "trial", trial), n_samples=cfg.n_samples,
               tol_rel=cfg.tol_rel, min_agree=cfg.min_agree, max_steps=cfg.max_steps,
               seeds=cfg.seeds, noise=cfg.noise)
    return c


def _gaps(all_rows: Dict, hard_rows: Dict) -> Dict:
    def gap(rows, a, b):
        if a not in rows or b not in rows:
            return None
        return round(rows[b]["f1"] - rows[a]["f1"], 3)
    return {
        "architecture_value_all": gap(all_rows, "Direct", "Hybrid"),
        "architecture_value_hard": gap(hard_rows, "Direct", "Hybrid"),
        "perception_cost_all": gap(all_rows, "Hybrid", "RuleSet"),
        "perception_cost_hard": gap(hard_rows, "Hybrid", "RuleSet"),
    }


def format_table(summary: Dict) -> str:
    lines = []
    method_rows = list(summary["all_components"].keys())
    header = (f"{'method':10s} {'prec':>6s} {'recall':>7s} {'F1':>6s} "
              f"{'F1 95% CI':>15s} {'probes':>7s} {'falseElim':>10s} "
              f"{'contain':>8s} {'|set|':>6s} {'F1@k':>6s}")

    def block(title, rows):
        lines.append(f"\n{title}")
        lines.append(header)
        for m in method_rows:
            r = rows[m]
            ci = f"[{r['f1_ci95'][0]:.3f},{r['f1_ci95'][1]:.3f}]"
            lines.append(f"{m:10s} {r['precision']:6.3f} {r['recall']:7.3f} "
                         f"{r['f1']:6.3f} {ci:>15s} {r['avg_probes']:7.2f} "
                         f"{r['avg_false_elims']:10.3f} {r['containment']:8.3f} "
                         f"{r['avg_set_size']:6.2f} {r['shortlist_f1']:6.3f}")

    block("== ALL components ==", summary["all_components"])
    block("== HARD subset (expert != {Injection_Molding}) ==", summary["hard_subset"])
    g = summary["diagnostic_gaps"]

    def fmt(v):
        return f"{v:+.3f}" if v is not None else "   n/a"
    lines.append("\ndiagnostic gaps (F1):")
    lines.append(f"  architecture value  Direct->Hybrid : "
                 f"ALL {fmt(g['architecture_value_all'])}   HARD {fmt(g['architecture_value_hard'])}")
    lines.append(f"  perception cost     Hybrid->RuleSet: "
                 f"ALL {fmt(g['perception_cost_all'])}   HARD {fmt(g['perception_cost_hard'])}")
    return "\n".join(lines)
