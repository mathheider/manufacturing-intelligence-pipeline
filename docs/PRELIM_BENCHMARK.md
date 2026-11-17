# The seeded benchmark and the first design (July 2026)

This is the README of the package as it stood for the preliminary exam. It documents the
seeded benchmark, the five pipelines and the first real-photo run. The current results are
in the top-level [README](../README.md).

A VLM-plus-symbolic-rules system that selects polymer-composite **manufacturing
processes** from component features. It is the runnable core of the preliminary
exam contribution, rebuilt from scratch as a clean, **zero-dependency,
fully-deterministic, tested** package.

The architecture separates the two halves of the decision:

```
        perception                            decision
   ┌──────────────────┐   committed      ┌────────────────────┐
   │  VLM (or MockVLM)│  features w/      │  symbolic rule     │   survivors
   │  reads the part  │─ uncertainty ───▶│  engine: 92 cited  │──▶ + cited
   │  only            │                  │  forbidden_if rules│   provenance
   └──────────────────┘                  └────────────────────┘
```

The VLM never sees the rules; the rule engine never sees the image. **Per-feature
provenance** binds them: every process elimination is traceable back to a
specific feature observation *and* the literature-cited rule that fired.

## The three-pillar contribution this demonstrates

1. **Empirical finding** — a direct RAG+LLM classifier over-predicts the
   high-frequency process (injection molding) even when it is wrong. Visible as
   the `Direct` pipeline's collapse on the HARD subset below.
2. **Architectural prescription** — VLM does perception only; a validated
   symbolic engine makes the categorical decision; provenance binds them. The
   `Hybrid` pipeline.
3. **Benchmark artifact** — 14 hand-labeled components (+ a deterministic
   synthetic expansion to any scale), a real CC-licensed photo pool with
   category-level expert labels, 17 processes, 92 cited rules (21 pending
   source verification, run `scripts/citation_status.py`), and a five-way
   ablation with the two diagnostic gaps. Every rule clause survived an
   adversarial counterexample review; 9 drafted clauses that did not were
   removed (see `docs/IMPROVEMENTS.md`).

## Quickstart (no dependencies, no API key)

```bash
python -m pip install -e .          # optional; or just set PYTHONPATH=src
mip demo                            # tiny offline demo
mip bench --seeds 40 --direct-topk auto   # the canonical benchmark
mip explain cf_cylinder             # one component, with cited provenance
mip rules                           # the rule table
python -m pytest                    # 44 tests, incl. locked golden numbers
```

Without an install: prefix commands with `PYTHONPATH=src python -m mip.cli …`.

## The five pipelines (the ablation)

| Pipeline     | Perception                    | Decision            | Role |
|--------------|-------------------------------|---------------------|------|
| **Direct**   | 1 VLM classify call           | VLM answers directly| RAG+LLM baseline; surfaces corpus bias |
| **Direct@k** | 1 VLM top-k call              | VLM answers directly| Set-size-matched baseline (k = Hybrid's mean survivor size) |
| **RuleSet**  | oracle (ground-truth) features| rule engine         | Perfect-perception **upper bound** |
| **Greedy**   | every feature once, noisy     | rule engine         | Ablation: no planning, no consistency |
| **Hybrid**   | goal-directed + consistency   | rule engine         | **The MIP agent — the contribution** |

Direct predicts a singleton while the elimination pipelines predict sets, so
plain set-F1 flatters Direct on an IM-heavy dataset. We report **both** the
singleton Direct and the set-size-matched `Direct@k` (goal-3 decision:
report both rather than replace).

## Canonical results (MockVLM, 14 components, 40 seeds, seed=20260630)

```
== ALL components ==
method       prec  recall     F1       F1 95% CI  probes  falseElim  contain  |set|   F1@k
Direct      0.523   0.445  0.468   [0.430,0.506]    1.00      1.120    0.389   1.00  0.468
RuleSet     0.234   1.000  0.360   [0.347,0.374]    0.00      0.000    1.000   7.86  0.324
Greedy      0.179   0.670  0.260   [0.243,0.276]   11.00      0.554    0.605   6.43  0.265
Hybrid      0.200   0.888  0.302   [0.286,0.318]    9.30      0.184    0.859   8.52  0.287
Direct@k    0.127   0.701  0.209   [0.199,0.219]    1.00      0.623    0.561   8.00  0.318

== HARD subset (expert != {Injection_Molding}) ==
method       prec  recall     F1       F1 95% CI  probes  falseElim  contain  |set|   F1@k
Direct      0.234   0.098  0.137   [0.110,0.164]    1.00      1.891    0.000   1.00  0.137
RuleSet     0.323   1.000  0.476   [0.462,0.490]    0.00      0.000    1.000   7.12  0.442
Greedy      0.220   0.556  0.299   [0.272,0.325]   11.00      0.834    0.444   5.33  0.271
Hybrid      0.274   0.839  0.394   [0.371,0.415]    9.49      0.287    0.787   7.10  0.375
Direct@k    0.129   0.477  0.199   [0.183,0.215]    1.00      1.091    0.231   8.00  0.181

diagnostic gaps (F1):
  architecture value  Direct->Hybrid : ALL -0.166   HARD +0.257
  perception cost     Hybrid->RuleSet: ALL +0.058   HARD +0.082
```

**How to read this.**
- **Corpus-frequency bias.** `Direct` scores a respectable F1 on ALL (most parts
  really are injection-molded) but **collapses on HARD** (F1 0.137, recall
  0.098) — it just answers "injection molding." That is the empirical failure
  mode, not a capability.
- **Architecture value** (`Direct → Hybrid`) is **+0.257 F1 on HARD** — the
  architecture pays off exactly where the bias hurts. Against the singleton
  Direct it is negative on ALL (a biased guess is often "right" on an
  IM-heavy set); against the **set-size-matched** `Direct@k` (k=8), Hybrid
  wins on **both** subsets (0.302 vs 0.209 ALL; 0.394 vs 0.199 HARD).
- **Consistency sampling** (`Greedy → Hybrid`) cuts false eliminations
  **~3×** (0.554 → 0.184 on ALL; 0.834 → 0.287 on HARD) using **fewer probes**
  (9.3 vs 11), because the planner stops once the decision is made and
  abstains on a feature after one failed re-probe (`max_reprobe=1`).
- **Perception cost** (`Hybrid → RuleSet`) is small (+0.082 F1 on HARD):
  calibrated perception recovers ~83% of the perfect-perception upper bound.
- **Elimination-native metrics:** `contain` = expert-containment rate (the
  full expert set survived), `|set|` = mean survivor-set size (from 17),
  `F1@k` = F1 of the top-3 survivors under the symbolic constraint-evidence
  ranking (`engine.shortlist`; the categorical call never goes back to the
  net). `RuleSet` containment is 1.0 by construction — the dataset invariant
  guarantees oracle features never eliminate an expert-feasible process.

## Reproducibility

Every number above is a pure function of `(code, config, seed)`. The offline
benchmark uses **no third-party packages** and is immune to `PYTHONHASHSEED`
(seeds derive from SHA-256, not the salted built-in `hash`). See
[REPRODUCIBILITY.md](../REPRODUCIBILITY.md) for the exact contract and
[IMPROVEMENTS.md](IMPROVEMENTS.md) for what changed from the original
prototype and why.

## Layout

```
mip/
├── pyproject.toml          installable; zero runtime deps; `mip` entry point
├── configs/default.json    every hyperparameter + master seed
├── data/                   frozen JSON the engine loads (rules, components)
├── scripts/build_data.py   canonical rule/feature/component source -> data/*.json
├── src/mip/
│   ├── rules.py            RuleSet: load + VALIDATE the rule table
│   ├── engine.py           pure rule engine, cited provenance
│   ├── state.py            ConstraintGraph / Elimination (structured output)
│   ├── vlm/                BaseVLM, MockVLM (deterministic), real backends
│   ├── sampling.py         consistency-calibrated commitment
│   ├── planner.py          goal-directed active perception
│   ├── pipelines.py        Direct / Direct@k / RuleSet / Greedy / Hybrid
│   ├── dataset.py          14 seeds + synthetic expansion + real photo pool
│   ├── metrics.py          P/R/F1, bootstrap CIs, elimination-native metrics
│   ├── benchmark.py        the deterministic ablation runner
│   └── cli.py              `mip bench|explain|rules|demo`
├── scripts/                image fetch/QC/coverage/citation tooling
├── assets/make_charts.py   results charts, values traceable to outputs/*.json
└── tests/                  43 tests incl. golden-number regression
```

## Real images

```bash
python scripts/fetch_images.py --per-category 70    # CC-licensed pool (resumable)
python scripts/qc_images.py --apply --workers 4     # model screen, cached
mip bench --dataset real --backend gemini --model gemini-2.5-flash-lite \
    --stratified --limit 150 --direct-topk 8 --workers 6 --confirm \
    --out outputs/benchmark_real.json
```

Real backends only affect *perception*; the rule engine, scoring, and provenance
are identical. Results with a live model are, by nature, not bit-reproducible —
the deterministic contract covers the offline MockVLM path; every raw response
is archived under `outputs/vlm_cache/` so real runs are resumable and auditable.

**Archived result (150 stratified CC photos, category-level labels, Flash-Lite,
prompt bench-v1):** the picture inverts. A 2026 commodity VLM ranks well
(Direct .510 ALL / .334 HARD; size-matched Direct@8 .427 HARD with .835
expert-containment) while feature-based elimination collapses (Hybrid .086
HARD, 2.29 false eliminations/part): single uncalibrated photos do not support
reliable feature reads. The per-feature provenance tally localizes 84% of the
false eliminations to two boolean reads (`hollow` 58%, from templated question
phrasing that reads open shells as hollow bodies; `continuous_fiber_required`
26%). The feature schema now carries explicit closed-hollow phrasing
(bench-v2) for the full-pool Flash run, which resumes across daily quota
windows (see REPRODUCIBILITY.md).
