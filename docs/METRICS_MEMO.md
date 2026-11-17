# Alternative Classification Metrics for the MIP Benchmark

**Date:** 2026-09-01
**Responds to:** committee action item "research and present alternative classification metrics"; a committee member's note that F1 is a poor fit and a confusion matrix would show class clustering.
**Evidence base:** `outputs/benchmark_real_v2.json` (n=635 real photos, preregistered) and the label distribution of the pool itself. Nothing here re-opens a preregistered headline; everything proposed is additive.

---

## 0. The finding that forces the change

A predictor that ignores the image entirely and always answers **Injection Molding** scores **set-F1 0.555** on the full real pool and **0.270** on the HARD subset.

| "Method" | ALL (n=635) | HARD (n=387) |
|---|---|---|
| Constant `Injection_Molding` (looks at nothing) | **0.555** | **0.270** |
| Top-3 by prevalence (looks at nothing) | 0.442 | 0.405 |
| Predict all 17 (looks at nothing) | 0.206 | 0.267 |
| Direct VLM | 0.605 | 0.383 |
| Greedy | 0.185 | 0.228 |
| Hybrid | 0.146 | 0.198 |
| Direct@k (k=8) | 0.352 | 0.436 |

Two degenerate predictors beat both pipeline variants on HARD, and one nearly matches the Direct VLM on the full pool. That is not a statement about the pipeline — it is a statement about the metric. **We have been reporting a number with no established floor.** A reviewer who computes this table will compute it in five minutes, and it is the single most likely reason for a desk-level "the baselines are missing" rejection.

This is also the honest version of the argument: better metrics are not a way to make Hybrid look better. They are a way to see *where* it fails.

---

## 1. Why the current metric is the wrong instrument

First a terminology correction for the notes: what we report is **not micro-F1**. It is per-instance set-F1 (predicted survivor set vs. expert-feasible set), macro-averaged over instances. Four problems:

**1.1 No informative floor.** Section 0. Prevalence alone buys 0.555.

**1.2 Prevalence dominates.** `Injection_Molding` is in 420 of 635 expert sets (66%); on HARD it is still in 172/387 (44%). The next most common label appears in 139. Any instance-averaged score is mostly a report on one class.

**1.3 It conflates two errors with opposite costs.** Set-F1 charges the same for (a) eliminating a process the expert says is feasible — a hard failure, the part cannot be made the way we now claim — and (b) carrying an extra survivor — a soft failure, the shortlist is one line longer. The elimination engine's whole job is safe narrowing; the metric prices safety and narrowing identically.

**1.4 It structurally rewards guessing.** Direct emits exactly one label (mean set size 1.000) against truths averaging 2.01 labels. Its recall is capped at ~0.50 by construction, but its precision is unpenalized for refusing to represent uncertainty. Hybrid emits 3.08 survivors and gets charged for every one it carries. Direct "wins" on set-F1 (0.605 vs 0.146) while being *incapable* of the task — its expert-set containment is 0.380, and on HARD it is **0.013**. Under containment, the ranking of the methods reverses. When a metric reverses the ranking, the metric is the finding.

**1.5 A plain confusion matrix is undefined.** 363 of 635 expert sets (57%) hold more than one process; mean truth size 2.01, HARD 2.65. There is no single true class to put on the row axis without a stated reduction rule. Section 3 gives three that are well-defined.

---

## 2. Proposed metric set

Tiered on purpose: Tier A is what the system is actually for, Tier B is what a reviewer needs to compare us to anything, Tier D is what makes any of it interpretable.

### Tier A — primary, elimination-native (decision-relevant)

| Metric | Definition | Status |
|---|---|---|
| **Expert-set containment** | fraction of instances where the full expert set survives (zero false elimination) | already computed |
| **Mean survivor-set size / reduction** | 17 − mean survivors; how far the space was narrowed | already computed |
| **Containment–size operating curve** | containment plotted against mean set size as the elimination threshold sweeps | **new — the headline figure** |
| **Per-process false-elimination rate** | for each process p, P(p eliminated \| p ∈ expert set) | **new** |

The operating curve is the right primary presentation and it is borrowed, deliberately, from conformal prediction, where a set-valued predictor is judged on **coverage** (does the true label survive) against **efficiency** (how small is the set). That framing already has a literature and referees accept it. It also dissolves the Direct-vs-Hybrid artifact: Direct is a single point at (size 1.0, containment 0.38), Direct@8 at (8.0, 0.88); Hybrid must be shown to dominate the curve those two points define, or it does not earn its complexity. That is a real, falsifiable claim — and on current numbers it is one we may fail, which is worth knowing before a reviewer tells us.

### Tier B — comparability (what a reviewer expects to see)

- **Per-class precision / recall / F1 with support** — label-based, one row per process. This is the table that replaces the single F1 and is the natural companion to the confusion matrix.
- **Macro-recall / balanced accuracy** — immune to the IM prevalence.
- **Multi-class MCC** — the standard answer to "F1 is misleading under imbalance"; unlike F1 it uses all four cells of the matrix and cannot be inflated by the majority class (Chicco & Jurman).
- **Jaccard (IoU), Hamming loss, subset accuracy** — the conventional multi-label triple; cheap to add, and reviewers look for them. Subset accuracy will be brutal (exact set match), which is why it is Tier B and not Tier A.
- **Cohen's κ** or an explicit **skill score** against the Tier D baseline (below).

### Tier C — ranking and calibration (where scores exist)

- **Containment@k / top-k accuracy** — Direct@k already instruments this.
- **Coverage error, ranking loss, LRAP** — standard for a ranked survivor list; we already rank via `engine.shortlist`.
- **ECE on the perception answers** — the calibration-first milestone is currently justified by an aggregate; this measures it directly and per feature (the 11 features), which tells us *which question* to fix next rather than that "perception is the constraint."

### Tier D — mandatory baselines (report beside every number)

Constant-IM, prevalence top-3, predict-all-17, uniform random-k, per-category majority. Then report the **skill score**

> skill = (score − baseline) / (ceiling − baseline)

with ceiling = the perfect-perception RuleSet oracle. We already do ceiling-normalization in the deck; adding the floor completes it, and it converts every headline from "0.198, is that good?" into a number that answers itself.

---

## 3. Confusion matrices that are actually defined

Three views, all computable from one per-instance record dump, all showing different clustering:

**3.1 Label co-occurrence matrix (17×17, set-native).** Cell (i, j) = number of instances where process i was in the expert set and process j was in the predicted set. Row-normalize. Diagonal = per-class recall; off-diagonal = which process is being substituted for which. Requires no reduction rule and is the direct answer to the committee's clustering question.

**3.2 Top-1 reduction (17×17, standard).** Reduce prediction to rank-1 survivor (Direct's single answer; Hybrid's top-ranked survivor under `engine.shortlist`) and truth to the expert set's rank-1 under a stated tie-break; score a hit if the prediction is anywhere in the expert set for a second, laxer variant. Gives the textbook matrix a committee reads instantly. Both variants get reported — the reduction rule is a choice and hiding it is how this kind of table misleads.

**3.3 Elimination-error matrix.** Rows = process wrongly eliminated, columns = process that survived in its place. This is the diagnostic that maps directly onto rule-set repair: a hot cell names the clause to fix.

**3.4 Family-level matrix (6×6).** Section 4. Coarse enough that the cells have counts worth reading — with 17 classes and 635 instances the fine matrix averages 2.2 instances per cell, and the tail classes (Structural RIM n=13, Thermoforming n=16, Filament Winding n=17) cannot support a stable rate at all.

---

## 4. Binary-first, then fine-grained (a committee proposal, made concrete)

Proposed two-level tree over the 17 processes:

| Family | Members | Label prevalence |
|---|---|---|
| Thermoplastic melt forming | Injection Molding, Blow Molding, Thermoforming, Overmolding | 456 |
| Compression (thermoset charge) | Compression Molding, SMC, BMC | 198 |
| Liquid composite molding | RTM, VARTM, Structural RIM | 152 |
| Additive | FFF, SLA, SLS, LOM | 55 |
| Open-mold composite | Hand Lay-up | 36 |
| Continuous profile | Pultrusion, Filament Winding | 19 |

Why this helps, measured on our own labels:

- **405 of 635 expert sets (64%) live entirely inside one family** (179 span two, 51 span three). At the process level only 272 (43%) are single-label. So the family level is a genuinely better-posed classification problem, not just a coarser one.
- **133 instances are multi-process but single-family** — those collapse to clean single-label instances at level 1. Much of what currently reads as "the model is wrong" is the model picking a sibling.
- A family-level confusion matrix has ~18 instances per cell instead of ~2.

**Level-0 binary.** Two candidates: (a) *fiber-reinforced vs unreinforced* — clean physics, aligns with the CCM framing, but strands Additive; (b) *net-shape tooling vs non-tooled* — cleaner separation for Additive and Hand Lay-up. Recommendation: report level 1 (6 families) as the primary coarse level and treat the binary as a presentation device, because a 6-way split is where our labels actually cluster. Worth 10 minutes of disagreement at the next meeting rather than a silent choice.

**Metrics at each level:** hierarchical precision/recall/F (hP, hR, hF — credit for ancestors correctly identified, so a sibling error costs less than a cross-family error), plus per-level containment. A sibling confusion and a cross-family confusion are not the same engineering failure and should not cost the same.

---

## 5. Per-process case studies

The pool has **38 part categories**, and the tail is thin: 15 categories have n < 10, six have n = 1. Per-category results at that resolution are noise. Design:

1. Break out by **family** (six groups above), with n and bootstrap CI on every cell.
2. Within a family, break out by category **only where n ≥ 25** — steering_wheel (70), dashboard (58), gear_shift_knob (42), instrument_cluster (37), rear_spoiler (34), headlamp_reflector (33), fender (30), electrical_junction_box (28), printed_bracket (27), printed_prototype (27). Ten categories, 386 of 635 photos.
3. The "all four variants agree" consistency check the committee asked for is a **rank-correlation across families**, not four separate bar charts.

**Honesty caveat that must ship with these:** real-pool expert labels are *category-level* feasible sets derived from the literature, not per-instance annotations (`dataset.load_real` docstring, `docs/IMPROVEMENTS.md`). A per-category breakout is therefore partly a report on the label definition. It must be stated in the caption, every time — otherwise the case-study section is the weakest point of the paper.

---

## 6. What it costs to compute

Every metric above except the Tier D baselines needs one thing we do not currently keep: **per-instance predictions**. `benchmark.run_benchmark` scores each instance and discards the prediction, writing only aggregates.

The fix is cheap and carries no scientific risk:

- Extend the existing cache-only replay (`scripts/score_v2_interim.py`, which already runs the frozen pipeline path with a backend that refuses network calls) to dump one record per instance × method: `{id, category, family, expert_set, predicted_set, ranked_survivors, n_probes, per_feature_answers, eliminated_by_clause}`.
- The v2 response archive holds **33,902 cached responses** for the full 635-photo run, so this is a **replay at zero API cost** and cannot perturb a preregistered number.
- Output is a new artifact (`outputs/records_real_v2.jsonl`); the preregistered headlines in `benchmark_real_v2.json` are never rewritten. Same additive discipline as the M-1 re-score.

Nothing in Section 2, 3, 4 or 5 can be produced before this record dump exists. It is the one blocking task.

---

## 7. Traceability

All figures cited: `outputs/benchmark_real_v2.json` (Hybrid HARD 0.198, CI 0.176–0.223; Direct 0.383; Greedy 0.228; Direct@k 0.436; n=635/387) and label statistics recomputed from `dataset.load_real()` on 2026-09-01. Baseline scores in Section 0 computed with the repo's own `metrics.score` against the same expert sets. The superseded interim readout is not cited anywhere in this memo.

**References:** Chicco & Jurman, *The advantages of the Matthews correlation coefficient over F1 and accuracy* (BMC Genomics, 2020); Sorower, *A literature survey on algorithms for multi-label learning* (Hamming loss / subset accuracy / Jaccard); Kiritchenko et al. and Silla & Freitas, hierarchical classification evaluation measures; Angelopoulos & Bates, conformal prediction (coverage vs. efficiency framing).
