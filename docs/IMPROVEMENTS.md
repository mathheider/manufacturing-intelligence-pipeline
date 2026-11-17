# Improvements over the original prototype

This rebuild replicates `mip_agent_prototype/` and improves each part. For every
module I asked: *what does this do, where does it fail, and how do I make it more
correct / reproducible / honest?* The answers are logged here so the changes are
auditable rather than silent.

## Summary table

| # | Part | Original | Problem | Improvement |
|---|------|----------|---------|-------------|
| 1 | Packaging | flat scripts, `from state import …` | breaks unless CWD is the folder; not importable/installable | proper `src/mip` package, relative imports, `pyproject.toml`, `mip` console script |
| 2 | Dependencies | hard `import yaml` for every run | offline benchmark needs a third-party package (and PyYAML wasn't even installed) | engine loads frozen **JSON**; **zero** runtime deps; YAML/real-VLM are optional extras |
| 3 | Seeding | `random.Random(seed)` sequential; `demo.py` gave each method a **different** seed | unfair comparison; `hash()`-style fragility; noise couples to call order | SHA-256-derived seeds (`_seed.py`); MockVLM noise = pure fn of `(seed, feature, index)`; every pipeline sees identical reads |
| 4 | `PYTHONHASHSEED` | implicit reliance on unsalted behavior | results could vary across processes | seeds never use built-in `hash`; verified identical under different hash seeds |
| 5 | Rule table | raw dict from `yaml.safe_load`, trusted | a typo, bad op, or bool-on-numeric threshold silently mis-prunes | `RuleSet` parses to typed dataclasses and **validates** feature names, ops, and value types at load |
| 6 | Provenance | citations were YAML **comments**; reason was a bare string | the "provenance binding" thesis wasn't in the data | citations are first-class `cite` + `reason` **data**; every elimination carries them; `mip explain` prints them |
| 7 | Rule engine | mutated a global graph in place | hard to test; order-dependent | pure `eliminate()` returns `(survivors, eliminations)`; a thin adapter mutates a graph for the agent |
| 8 | bool/int ops | `value > target` with Python `bool` (an `int` subclass) | `True > 0.5` silently truthy | ordered ops guarded against bool operands |
| 9 | Pipeline naming | `single_pass` / `greedy` / `agent` | drifted from the locked story's `Direct / Rule Set / Hybrid`; no true oracle "Rule Set" pipeline | renamed to **Direct / RuleSet / Greedy / Hybrid**; added the real **RuleSet-on-oracle** upper bound |
| 10 | Diagnostic gaps | described in prose only | not computed | benchmark computes **architecture value** (Direct→Hybrid) and **perception cost** (Hybrid→RuleSet), ALL and HARD |
| 11 | Consistency (numeric) | `rel = stdev/mean`; unanimous **zero** reading → `rel = inf` → skipped | agent could never commit a genuinely-zero feature (e.g. draft = 0°), so it missed valid eliminations | perfect agreement (`stdev ≤ abs_tol`) is stable regardless of mean; recovered eliminations, raised Hybrid F1 |
| 12 | Dataset scale | 14 hand-labeled; "1000" only via Wikimedia scrape + paid Gemini | the headline "1,000-instance benchmark" was not offline-reproducible | deterministic synthetic expansion to any N, feasibility-invariant enforced by the oracle engine |
| 13 | Metrics | point estimates only | handoff lists "printed confidence interval" as an open TODO | seeded **bootstrap 95% CIs** on macro-F1 |
| 14 | Tests | none | no guard against regressions; numbers unverifiable | 32 tests: engine, validation, dataset invariant, sampling/mock determinism, pipelines, **golden-number regression** |
| 15 | Config | hyperparameters scattered across files | can't reproduce a run from its output | one `Config` dataclass, a JSON config file, and the config **echoed into every result** |
| 16 | Output | prose summary | not machine-checkable | JSON-serialisable `ConstraintGraph` with per-feature observations, cited eliminations, and a trace |

## The three that matter most

**#3 + #4 — the reproducibility backbone.** The original `demo.py` seeded the
agent with 42, single-pass with 123, and greedy with 7 — so the three methods
were compared on *different* perception-noise realizations. Any F1 difference
mixed method quality with luck. Here a single per-instance seed feeds all
pipelines, and noise is a pure function of `(seed, feature, index)`, so the
comparison is a genuine controlled experiment. Combined with SHA-256 seeding,
the whole benchmark is byte-reproducible across machines and hash seeds.

**#6 — provenance as data.** The prelim's central novelty claim is *per-feature
provenance binding perception to decision*. In the original, the binding lived in
YAML comments and ad-hoc reason strings. Promoting `cite`/`reason` to structured
fields makes each elimination an auditable record — `mip explain cf_cylinder`
prints, for every cut process, the observed feature, the rule that fired, and its
literature source. The thesis is now demonstrable from the artifact itself.

**#11 — a real correctness fix found by looking.** Running `mip explain` showed
the agent skipping a unanimous `draft_angle = [0,0,0,0,0]` because relative
standard deviation divides by a zero mean. A physically-zero feature could never
be committed, so the agent left injection molding in the candidate set for the
carbon-fiber cylinder when the oracle correctly removed it. Treating perfect
agreement as stable fixed it: Hybrid now matches the oracle exactly on that part,
HARD-subset F1 rose 0.360 → 0.367 (regression config), and the perception-cost
gap shrank.

## #17 — Phase-1/2 upgrade (2026-07-01): densified rules, elimination-native metrics, real-image path

**Golden numbers re-locked in this change** (allowed: intentional + documented).
Canonical 40-seed run moved from `RuleSet F1 .290/.390 (ALL/HARD)` to
`.360/.476`, `Hybrid .272/.349` to `.314/.414`. Why, in order of effect:

1. **Three discriminative features added** (goal 1): `max_dimension_mm`
   (build-envelope checks), `production_volume` (ordinal 0/1/2:
   prototype / low / high), `continuous_fiber_required`. All 14 seed
   components hand-assigned values; each assignment checked against every
   expert process's clause set, then confirmed by an independent review pass.
   `surface_class` (optional in the goals) was NOT added — the three above
   already double oracle precision, and each extra feature costs ~n_samples
   VLM calls per image on the real benchmark.
2. **Rule table 71 → 92 cited clauses** (goal 2). 30 clauses were drafted
   against the RULE POLICY; every draft then went through an **adversarial
   counterexample review** (six independent domain reviewers, 2026-07-01),
   which REFUTED 9 drafts — e.g. "RTM infeasible at mass volume" fails
   against HP-RTM (BMW i3 series production), "overmolding cannot deliver
   continuous fiber" was backwards (organosheet insert overmolding is the
   industrial route TO continuous-fiber parts), and "SLA/SLS infeasible at
   mass volume" fails against Align Technology's and Erpro's print farms.
   7 thresholds were raised to documented machine envelopes (e.g. FFF 1200 →
   2200 mm, Tractus3D T3500; SLS 800 → 1000 mm, Farsoon HT1001P). The
   refuted drafts are kept as NOTE comments in `scripts/build_data.py` so the
   review is auditable. Consequence: the artifact has **92** cited clauses,
   not the "100" the old deck claimed — deck/report must say 92 (and that 21
   of them carry `[VERIFY:` proposed citations pending Mathias's source
   check; `scripts/citation_status.py` prints the table).
3. **Set-size asymmetry rebalanced** (goal 3): Direct predicts a singleton
   vs Hybrid's ~8-element survivor set, which flatters Direct on an IM-heavy
   set. Decision: **report both** — the singleton `Direct` and a
   set-size-matched `Direct@k` (`--direct-topk auto` sets k to Hybrid's mean
   survivor size, k=8 on the canonical run). Size-matched, Hybrid wins both
   subsets (ALL .314 vs .209; HARD .414 vs .199).
4. **Elimination-native metrics** (goal 4): expert-containment rate, mean
   survivor-set size / reduction from 17, and shortlist F1@k under a purely
   symbolic constraint-evidence ranking (`engine.shortlist`: count of a
   survivor's clauses whose features were observed and passed; the
   categorical call never goes back to the net).
5. **Config unchanged** (n_samples=5, min_agree=0.6): stricter/wider
   alternatives (min_agree=0.8; n_samples=7) were measured and traded recall
   for precision without improving HARD F1 or the perception-cost gap.
6. **Real-image path** (goal 8 wiring): `dataset.load_real` (category-level
   expert labels, QC-accepted images only), `vlm/cache.py` disk cache of
   every raw response keyed (image sha256, pipeline:channel, sample index,
   prompt version) — resumable, auditable, free re-runs — and a CLI cost
   guardrail (`--confirm` after a printed call estimate, goal 9). The
   RuleSet (oracle) pipeline is undefined on photos (no ground-truth
   features) and is skipped for `--dataset real`.

## #18 — Probe budget (2026-07-01): abstain after one failed re-probe

**Golden numbers re-locked again in this change** (Hybrid ALL .318 → .304,
HARD .416 → .399 on the 5-trial regression config; canonical 40-seed HARD
.414 → .394). The 20-image real-backend smoke run exposed the cost: when a
feature failed the consistency check, the planner re-picked that same feature
every step until `max_steps`, so a real image cost ~98 VLM calls instead of
~68, with no decision change (a systematically mis-read photo feature does
not stabilize by asking harder). The agent now gives an unstable feature
`max_reprobe` retries (default 1) and then abstains on it. The mock benchmark
pays a small F1 cost for this honesty (unbiased mock noise DID stabilize
under infinite retries; biased real errors do not), and the bounded budget is
what makes the full real-image run affordable. Both the old and new operating
points are documented here so the change is auditable.

## Honesty notes (per CLAUDE.md "never fabricate")

- The synthetic 1000-instance set is **controlled synthetic data**, not a new
  expert survey. Its expert labels are inherited from the 14 archetypes; only
  geometry is varied, and always within the rule-feasible envelope. It measures
  behavior at scale; it does not add ground truth.
- `RuleSet` recall is 1.0 **by construction** (the dataset invariant guarantees
  oracle features never eliminate an expert process). It is an upper bound on
  perception, not a claim that the rules uniquely identify the right process —
  they are deliberately conservative, hence its low precision.
- Real-VLM (`--backend gemini|claude`) runs are not bit-reproducible; the
  deterministic contract is explicitly scoped to the offline MockVLM path.
- The real-image pool's expert labels are **category-level** sets (the
  processes feasible for the part category, from `scripts/image_categories.py`,
  derived from standard practice + LIT_REVIEW.md and flagged `[VERIFY]` where
  newly proposed) — NOT per-instance expert annotations. Never call them
  "expert ground truth of each image."
- The QC pass is a **model screen** (Gemini, prompt v2, cached); the
  30-image `outputs/qc_sample_sheet.html` exists so a human can audit what
  the screen accepted.
