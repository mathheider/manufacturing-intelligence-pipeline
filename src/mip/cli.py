"""Command-line interface.

    mip bench    [--size N] [--dataset seed|synthetic|real] [--backend mock|gemini|claude]
                 [--seeds K] [--direct-topk k|auto] [--limit N] [--workers W]
                 [--confirm] [--graphs]
    mip explain  <component-id>          # run the Hybrid agent, print provenance
    mip rules                            # rule-table summary
    mip demo                             # tiny offline demo, 4 archetypes

Cost guardrail (goal 9): any run with a real backend prints an API-call
estimate (minus what the disk cache already holds) and refuses to spend
without an explicit ``--confirm``. MockVLM remains the deterministic
canonical contract; real-image runs are the robustness result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .benchmark import format_table, run_benchmark
from .config import Config
from .dataset import load_seed, load_real
from .pipelines import run_hybrid, run_ruleset
from .rules import RuleSet
from .vlm import MockVLM
from ._seed import derive_seed

_OUT = Path(__file__).resolve().parents[2] / "outputs"


def _estimate_real_calls(args, cfg: Config) -> tuple:
    """(n_images, est_total_calls, cached_files) for a real-backend run."""
    rules = RuleSet.load()
    n_feat = len(rules.features)
    if args.dataset == "real":
        images = load_real()
        if args.limit:
            images = images[:args.limit]
        n_images = len(images)
    else:
        root = _OUT.parent
        n_images = sum(1 for i in load_seed()
                       if i.image and (root / i.image).exists())
    per_image = 1                                   # Direct
    if args.direct_topk not in (None, "", "0", 0):
        per_image += 1                              # Direct@k
    per_image += n_feat                             # Greedy, one read each
    per_image += cfg.n_samples * min(cfg.max_steps, n_feat)   # Hybrid ceiling
    cache_dir = _OUT / "vlm_cache" / "bench"
    cached = sum(1 for _ in cache_dir.glob("*.json")) if cache_dir.exists() else 0
    return n_images, n_images * per_image, cached


def _cmd_bench(args) -> int:
    cfg = Config.load(args.config)
    if args.seeds is not None:
        cfg.seeds = args.seeds
    if args.n_samples is not None:
        cfg.n_samples = args.n_samples
    topk = -1 if str(args.direct_topk).lower() == "auto" else int(args.direct_topk or 0)

    if args.backend != "mock":
        n_images, est, cached = _estimate_real_calls(args, cfg)
        print(f"real-backend run: {n_images} images, <= {est} VLM calls "
              f"(ceiling; actual is lower because the Hybrid agent stops early), "
              f"{cached} responses already cached in outputs/vlm_cache/bench")
        if not args.confirm:
            print("refusing to spend API calls without --confirm", file=sys.stderr)
            return 2

    summary = run_benchmark(cfg, size=args.size, backend=args.backend,
                            n_boot=args.n_boot, save_graphs=args.graphs,
                            out_path=args.out, dataset=args.dataset,
                            limit=args.limit, direct_topk=topk,
                            shortlist_k=args.shortlist_k, workers=args.workers,
                            model=args.model, stratified=args.stratified,
                            rules_path=args.rules)
    print(f"backend={summary['backend']}  dataset={summary['dataset']}  "
          f"instances={summary['dataset_size']}  "
          f"trials={summary['trials']}  seed={cfg.seed}")
    print(format_table(summary))
    if "vlm_cache" in summary:
        c = summary["vlm_cache"]
        print(f"\nvlm cache: {c['hits']} hits, {c['misses']} paid calls")
    print(f"\nwrote {summary['_out_path']}")
    return 0


def _cmd_explain(args) -> int:
    rules = RuleSet.load()
    cfg = Config.load(args.config)
    seed_set = {i.id: i for i in load_seed()}
    if args.component not in seed_set:
        print(f"unknown component {args.component!r}. options: "
              f"{', '.join(sorted(seed_set))}", file=sys.stderr)
        return 2
    inst = seed_set[args.component]
    vlm = MockVLM(noise=cfg.noise, seed=derive_seed(cfg.seed, inst.id))
    res = run_hybrid(vlm, inst.features, rules, cfg, verbose=True)
    oracle = run_ruleset(inst.features, rules)
    print(f"\ncomponent : {inst.id}  ({inst.image})")
    print(f"expert    : {sorted(inst.expert_processes)}")
    print(f"Hybrid    : {sorted(res.predicted)}  ({res.extra.get('committed')} features committed)")
    print(f"RuleSet   : {sorted(oracle.predicted)}  (oracle / perfect perception)")
    print("\neliminations (cited provenance):")
    for proc, elim in sorted(res.graph.eliminations.items()):
        print(f"  - {proc}: {elim.as_text()}")
    if args.json:
        Path(args.json).write_text(json.dumps(res.graph.to_dict(), indent=2),
                                   encoding="utf-8")
        print(f"\nwrote graph -> {args.json}")
    return 0


def _cmd_rules(args) -> int:
    rules = RuleSet.load()
    print(rules)
    for proc in rules.processes:
        clauses = rules.clauses_for(proc)
        print(f"\n{proc}  ({len(clauses)} clauses)")
        for c in clauses:
            print(f"  {c.feature} {c.op} {c.value}   [{c.cite}]")
    return 0


def _cmd_demo(args) -> int:
    cfg = Config(seeds=8)
    summary = run_benchmark(cfg, size=0, backend="mock", n_boot=500)
    print("MIP offline demo (14 seed components, MockVLM, 8 trials)")
    print(format_table(summary))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mip", description="Manufacturing Intelligence Pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bench", help="run the ablation benchmark")
    b.add_argument("--size", type=int, default=0, help="0=14 seed comps; else synthetic N")
    b.add_argument("--dataset", default="seed", choices=["seed", "synthetic", "real"],
                   help="'real' = the CC-licensed photo pool (needs a real backend)")
    b.add_argument("--backend", default="mock", choices=["mock", "gemini", "claude"])
    b.add_argument("--seeds", type=int, default=None, help="MockVLM trials to average")
    b.add_argument("--n-samples", dest="n_samples", type=int, default=None)
    b.add_argument("--n-boot", type=int, default=1000, help="bootstrap resamples for CIs")
    b.add_argument("--direct-topk", default=0,
                   help="Direct emits its top-k set; int or 'auto' (= Hybrid's mean survivor size, mock only)")
    b.add_argument("--shortlist-k", type=int, default=3,
                   help="k for the symbolic shortlist F1@k metric")
    b.add_argument("--limit", type=int, default=0,
                   help="cap real-dataset images (smoke runs)")
    b.add_argument("--workers", type=int, default=1,
                   help="parallel images for real backends")
    b.add_argument("--model", default=None,
                   help="override the backend model (gets its own cache dir)")
    b.add_argument("--stratified", action="store_true",
                   help="interleave real-dataset categories so --limit N is balanced")
    b.add_argument("--confirm", action="store_true",
                   help="required to spend real API calls")
    b.add_argument("--graphs", action="store_true", help="also dump Hybrid provenance graphs")
    b.add_argument("--config", default=None, help="path to a config JSON")
    b.add_argument("--rules", default=None,
                   help="alternate rules JSON (robustness ablations; default data/rules.json)")
    b.add_argument("--out", default=None, help="summary output path")
    b.set_defaults(func=_cmd_bench)

    e = sub.add_parser("explain", help="run Hybrid on one component with provenance")
    e.add_argument("component")
    e.add_argument("--config", default=None)
    e.add_argument("--json", default=None, help="write the constraint graph to this path")
    e.set_defaults(func=_cmd_explain)

    r = sub.add_parser("rules", help="print the rule table")
    r.set_defaults(func=_cmd_rules)

    d = sub.add_parser("demo", help="quick offline demo")
    d.set_defaults(func=_cmd_demo)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
