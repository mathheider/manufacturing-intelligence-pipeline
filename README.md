# Manufacturing Intelligence Pipeline (MIP)

MIP looks at a photo of a polymer or composite part and ranks the 17 manufacturing
processes that could have made it: injection molding, compression molding, resin
transfer molding, filament winding, 3D printing and others. A vision-language model
(VLM) reads the part. A rule engine written from manufacturing textbooks and papers
turns what the model read into a ranking, and shows which rules moved each process up
or down.

This repository holds the code, both rule bases, the benchmarks and every result file
behind the numbers below. It is the code of my Ph.D. work at the University of Delaware
(preliminary exam defended in August 2026). It covers perception and decision on photos
of single parts. The stage that detects and segments the parts in a photo of a whole
vehicle is not in this repository.

## Results on real photos (October 2026)

The benchmark is 635 openly licensed photos of real parts in 41 part categories, each
screened for quality. The correct processes are labeled by part category, about two per
part. 535 photos are for development. 100 are held out (Gold-100) and are scored once
per frozen model. Every number is a percentage of photos:

- **Top pick correct**: the process ranked first is one of the correct processes.
- **Right family**: the process ranked first is in a correct process family (melt,
  compression, liquid molding, open mold, continuous, additive).
- **All correct in top 3** and **all correct in top 8**: every correct process for the
  part is in the top 3, or the top 8.

### Held-out test (Gold-100, 100 photos, scored once)

| Method | Top pick correct | Right family | All correct in top 3 | All correct in top 8 |
|---|---|---|---|---|
| Guessing the most common processes, without the photo | 58% | 68% | 28% | 56% |
| VLM alone (Gemini 2.5 Flash-Lite, its own ranking) | 74% | 83% | 47% | 88% |
| First design: 92 rules as vetoes | 19% | 32% | 5% | 11% |
| The same 92 rules as evidence, with the VLM | 72% | 83% | 46% | 87% |
| All 564 rules as evidence, with the VLM | 73% | 83% | 48% | 88% |
| The 142 rules that pass the direction check, with the VLM | 72% | 81% | 45% | 90% |
| The 142 checked rules alone, no VLM ranking | 36% | 73% | 39% | 70% |

All settings were chosen on the development photos and frozen, with file hashes, before
Gold-100 was scored ([docs/RULES_V3_PLAN.md](docs/RULES_V3_PLAN.md)).

1. **Rules that delete processes fail on real photos.** In the first design one misread
   feature removed the right answer. Its top pick is correct on 19% of test photos,
   below the 58% from guessing the most common processes without looking.
2. **The same rules used as evidence work.** On top of the VLM's own ranking they reach
   72% to 73%, level with the VLM alone (74%), and every ranking comes with
   the rules behind it. Against the VLM alone, all 564 rules change the top pick by
   −1 point (95% range −3 to 0) and all-correct-in-top-3 by
   +1 point (95% range −3 to +6), with ranges from resampling the photos. On 100 photos
   the two cannot be told apart.
3. **The rules carry information of their own.** With no VLM ranking at all, the 142
   checked rules put the top pick in the right family on 73% of test photos (guessing:
   68%) and every correct process in the top 3 on 39% (guessing: 28%).

The same table for the development photos and for the hard photos (parts where
injection molding is not the only answer) is in
[outputs/soft_v3_summary_percent.json](outputs/soft_v3_summary_percent.json).

### Learned trust per feature (cross-validation on the 535 development photos)

Instead of hand-set weights, the development photos set one trust number for each of the
11 perceived features, plus one for rules that combine features: about 22 learned
numbers. The hand-written rules keep their direction, and there are no per-process base
rates, so the model cannot memorize part types. Every photo is scored by a model that
never saw it.

| Model | Folds | Top pick correct | Right family | All correct in top 3 | All correct in top 8 |
|---|---|---|---|---|---|
| VLM alone | both | 77.0% | 83.7% | 55.1% | 88.2% |
| VLM signals only, learned, no rules | new photos | 77.4% | 84.5% | 55.5% | 89.0% |
| One trust weight per rule | new photos | 78.3% | 84.5% | 60.7% | 93.5% |
| **One trust weight per feature (the frozen model)** | new photos | 77.6% | 84.5% | 57.2% | 91.2% |
| VLM signals only, learned, no rules | unseen part types | 77.4% | 84.7% | 55.7% | 89.0% |
| One trust weight per rule | unseen part types | 76.4% | 84.1% | 51.0% | 84.7% |
| **One trust weight per feature (the frozen model)** | unseen part types | 77.4% | 84.3% | 57.0% | 90.8% |

- **Trust per feature beats the VLM alone at keeping every correct process in the
  top 8**: +3.0 points (95% range +1.1 to +4.9) on new photos, and
  +2.6 points (95% range +0.9 to +4.5) on part types never seen in training.
  Top 3 rises by about 2 points, which is not yet significant:
  +2.1 points (95% range −0.2 to +4.3) and +1.9 points (95% range −0.6 to +4.5).
  The top pick does not change.
- A weight per rule looks better on familiar part types, where top 3 changes by
  +5.6 points (95% range +2.8 to +8.8), and loses on unseen ones, where it changes by
  −4.1 points (95% range −7.1 to −1.3). It memorizes part types, so it is not the
  frozen model.
- The learned trust is highest for production volume (0.53) and undercuts (0.29), and
  zero for wall thickness, draft angle and projected area. That matches the separate
  rule check: the VLM's guess of production volume is useful although volume cannot be
  seen, and its wall-thickness reads point the wrong way on real photos.

The frozen model is [outputs/trust_v3_frozen.json](outputs/trust_v3_frozen.json). It has
not been scored on Gold-100 yet. That single scoring is planned together with per-photo
labels, and `scripts/trust_v3.py --test` refuses a second one.

### A small open model, fine-tuned (Gold-100, scored once)

SmolVLM-256M with LoRA, trained on the 535 development photos on a CPU, with a recipe
fixed before any result ([docs/FINETUNE_V1.md](docs/FINETUNE_V1.md)).

| Model | Top pick correct | Right family | All correct in top 3 | All correct in top 8 |
|---|---|---|---|---|
| Guessing the most common processes | 58% | 68% | 28% | 56% |
| Gemini 2.5 Flash-Lite alone, prompted | 74% | 83% | 47% | 88% |
| SmolVLM-256M, prompted | 59% | 68% | 29% | 40% |
| **SmolVLM-256M + LoRA, fine-tuned** | 56% | 68% | 43% | 73% |

Against the same model without fine-tuning, all correct in top 8 changes by
+33 points (95% range +22 to +44) and top 3 by +14 points (95% range +5 to +23).
Fine-tuning teaches the small model which processes belong together. It does not teach
it to see manufacturing: its top pick does not improve, and on part types it never saw
it is at the level of guessing.

## How the design got here

Earlier results use set-F1 (the overlap between the predicted and the correct set of
processes) on the hard cases, where injection molding is not the only answer.

| Benchmark | VLM asked directly | First design (rules as vetoes) |
|---|---|---|
| Seeded: 1,000 instances built from 14 expert-labeled components, simulated perception | 0.136 | 0.385 (rules with perfect features: 0.510) |
| Real photos, first run: 150 photos (103 hard) | 0.334 | 0.086 |
| Real photos, preregistered: 635 photos (387 hard) | 0.383 | 0.198 (95% CI 0.176 to 0.223) |

- **Simulation.** With simulated perception noise and a direct classifier that carries
  an injection-molding prior, the pipeline recovered about three quarters of what the
  rules achieve with perfect features. Details: [docs/PRELIM_BENCHMARK.md](docs/PRELIM_BENCHMARK.md).
- **Real photos.** The picture inverted. The record of which observation fired which
  rule showed why: on the first 150 photos, 84% of the wrong eliminations came from two
  yes/no reads, whether the part is hollow (58%) and whether it needs continuous
  fiber (26%).
- **Preregistration.** The hypotheses for the 635-photo run, and the line that would
  count as success (hard-case F1 of 0.20), were committed before the run. The result was
  0.198: a real improvement over 0.086, and under the line. It is reported as committed in
  [docs/PREREGISTRATION_BENCH_V2.md](docs/PREREGISTRATION_BENCH_V2.md).
- **The metric had no floor.** A predictor that always answers injection molding scores
  set-F1 0.555 on all 635 photos and 0.270 on the hard ones
  ([docs/METRICS_MEMO.md](docs/METRICS_MEMO.md)). The four percentages above, reported
  next to a guessing baseline, replaced it.
- **The model's bias.** Asked for one answer, the VLM named injection molding for
  107 of 150 photos (71%). In a survey, experts chose it in 24% of their picks
  across 42 parts. These are different parts, so the shares are compared, not paired
  answers. On the same 150 photos the VLM's top-8 list held every correct process for
  133, against 6 for the veto pipeline.

## Run it

Python 3.9 or newer. The engine, the offline benchmark and the tests need no third-party
packages and no API key.

```bash
python -m pip install -e .                 # optional; or set PYTHONPATH=src
python -m pytest                           # 44 tests, with locked reference numbers
mip bench --seeds 40 --direct-topk auto    # the seeded benchmark, deterministic
mip explain cf_cylinder                    # one component, with the cited rules that fired
mip rules                                  # the 92-rule table
```

The October results are recomputed from the archived model responses, with no API calls:

```bash
python scripts/eval_soft_v3.py             # rules as evidence, development photos
python scripts/summary_percent_v3.py       # the percentage tables with 95% ranges
python scripts/rule_validity_v3.py         # which rules point the right way on development photos
pip install numpy scipy
python scripts/trust_v3.py --refreeze      # cross-validation of the learned trust model, about 30 s
```

`--test` on `eval_soft_v3.py` and `trust_v3.py` scores the held-out Gold-100 photos.
Please leave that for a new, frozen model.

Running the VLM yourself needs the photos (see below) and a Gemini API key in the
environment: `pip install -e ".[gemini]"`, then the commands in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## What is in the repository

```
src/mip/            the package: rule loading and validation, the veto engine (engine.py),
                    the evidence scorer (evidence.py), VLM backends, planner, pipelines,
                    dataset, metrics, benchmark runner, command line
data/               rules.json (92 cited veto rules), rules_v3.json (564 rules as evidence),
                    rules_v3_review.csv (each rule with how often it points the right way),
                    components.json (the 14 expert-labeled components)
scripts/            building the rule bases, fetching and screening photos, evaluation,
                    the learned trust model, fine-tuning, figures
tests/              44 tests, including the reference numbers of the seeded benchmark
outputs/            every result file behind the numbers above, and the archived VLM
                    responses (41,420 of them) as one archive
images_pool/        the photo manifests: source, author and license of every photo
docs/               the preregistration, the Gold-100 protocol, the metrics memo,
                    the rules-v3 plan and results, the fine-tuning report
```

## Data and licenses

- **Photos.** The photos are not in the repository. `images_pool/images_manifest.json`
  lists each one with its source page, author and license (Creative Commons or public
  domain, from Wikimedia Commons and Openverse), and `scripts/fetch_images.py` downloads
  them again. `images_pool/gold100_manifest.json` lists the 100 held-out photos.
- **Expert survey.** `outputs/survey_stats.json` holds the aggregated statistics, with
  rater identifiers replaced. The row-level export is not published because it carries
  respondent metadata.
- **Rules.** Each of the 92 veto rules cites its source. The 564 evidence rules cite
  works from the same reference base by topic and are marked as drafts until each
  citation is checked.
- **Code.** MIT license, see [LICENSE](LICENSE).

## Limits

- Labels on the real photos are by part category, not by photo. Per-photo labels for
  Gold-100 are in progress and will change the numbers.
- Development and test photos share part categories, so Gold-100 measures new photos,
  not new kinds of parts. The cross-validation with unseen part types covers that case.
- On 100 photos, differences of a few points at the top are within noise.
- The rules-as-evidence engine is level with the VLM alone on the top pick, not ahead.
  The limit is what the VLM can read from a photo. The next step asks about marks a
  camera can see (a pinch-off seam, layer lines, ejector-pin marks), with the success
  criterion written down before any data ([docs/RULES_V3_PLAN.md](docs/RULES_V3_PLAN.md), section 6).

## Contact

Mathias Heider, University of Delaware. Site: <https://mathheider.github.io>
