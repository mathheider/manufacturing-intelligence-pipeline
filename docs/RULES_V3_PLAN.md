# Rules v3: Expansion Plan and First Results

**Date:** 2026-10-12. **Status:** phases 1-5 executed on the archived bench-v2 responses (zero API calls); phase 6 (visible-signature perception pass) is built, tested with a mock backend, and ready to run.
**Frozen and untouched:** `data/rules.json` (92 clauses), `src/mip/engine.py`, `benchmark_real_v2.json`, the preregistered bench-v2 verdicts. Everything below is additive, in new files.

---

## 1. What changed

| | v2 (frozen) | v3 (this work) |
|---|---|---|
| Rules | 92 `forbidden_if` clauses | **564 rules** (6.1x): 223 favor, 249 disfavor, 92 forbid |
| Perception features | 11 | **39** (11 existing + 28 visible manufacturing signatures) |
| Decision | hard veto: one firing clause deletes a process | **soft evidence**: each rule moves a process's score, scaled by how strongly the perception samples agree; nothing is deleted |
| VLM role | perception only | perception + its own ranked top-8 used as a prior |
| Rule quality control | citation + adversarial review | same, **plus a validity check on real photos**: does the rule's evidence point the right way? |
| Provenance | first clause that eliminated each process | every contribution, with rule id and citation, per process |

Rule groups in `data/rules_v3.json` (reviewable as `data/rules_v3_review.csv`):

| Group | What | Rules |
|---|---|---|
| G1 | the 92 v2 limits, kept verbatim as strong soft evidence (weight 3) | 92 |
| G2 | process x feature typicality matrix on the 11 existing features | 118 |
| G3 | combinations of 2-3 features (e.g. closed + axisymmetric -> blow molding, filament winding) | 60 |
| G4 | **visible manufacturing signatures**: ejector-pin marks, gate vestige, parting line, pinch-off seam, woven fabric, chopped mat, helical winding pattern, layer lines, raster beads, powdery SLS skin, support nubs, thinned corners, one-sided finish, two bonded materials, sink marks, knit lines, molded clips, flash, gelcoat, sandwich core and more (28 features) | 269 |
| G5 | fiber architecture, both directions (continuous vs chopped) | 25 |

Citation policy: every v3 rule cites works already in the v2 reference base, by topic, marked `v3-draft [VERIFY]`. No page numbers are claimed for v3 rules yet.

## 2. Plan and status

| # | Phase | Status |
|---|---|---|
| 1 | Map archived responses to photos without the image files (`image_sha24_v2pool.json`); extract every read per photo and feature (`evidence_v2pool.json`, 41,420 reads; top-8 lists match the records exactly) | done |
| 2 | Observability audit of the 92 v2 clauses | done (section 4) |
| 3 | Soft-evidence scorer (`src/mip/evidence.py`) | done |
| 4 | Expansion to 564 rules + 28 signature questions (`scripts/build_rules_v3.py`) | done, drafts |
| 5 | Dev evaluation (535 photos), rule validity check, configs frozen (hashes below), Gold-100 test scored once | done (section 3) |
| 6 | Visible-signature perception pass (`scripts/perceive_signatures_v3.py`, one batched call per photo, 3 samples, ~1,905 calls) | **ready to run** (needs a Gemini API key) |
| 7 | Re-run validity + evaluation with signatures (same protocol, section 6) | after 6 |
| 8 | Verify citations and run the adversarial counterexample pass, validated rules first | to do |
| 9 | Grow further only where validity holds | to do |

## 3. Results so far (current 11 features only; visible marks not yet perceived)

**How to read the numbers.** Every number is a percentage of photos (out of 100; higher is better):

- **Top pick correct**: the process ranked #1 is one of the correct processes for that part.
- **Right family**: the process ranked #1 is in one of the correct process families (melt, compression, liquid molding, open mold, continuous, additive).
- **All correct in top 3**: every correct process for that part appears in the top 3.
- **All correct in top 8**: every correct process appears in the top 8.

A part usually has about two correct processes. Settings were chosen on the 535 dev photos and frozen before the 100 Gold-100 test photos were scored.

**Test: Gold-100, held out (100 photos)**

| Method | Top pick correct | Right family | All correct in top 3 | All correct in top 8 |
|---|---|---|---|---|
| Guess the most common processes (ignores the photo) | 58% | 68% | 28% | 56% |
| VLM alone (its own top-8 ranking) | 74% | 83% | 47% | 88% |
| Current Hybrid (rules delete processes) | 19% | 32% | 5% | 11% |
| Original 92 rules as evidence + VLM | 72% | 83% | 46% | 87% |
| All 564 rules + VLM | 73% | 83% | 48% | 88% |
| 142 checked rules + VLM | 72% | 81% | 45% | 90% |
| 142 checked rules alone (no VLM) | 36% | 73% | 39% | 70% |

**Dev (535 photos)**

| Method | Top pick correct | Right family | All correct in top 3 | All correct in top 8 |
|---|---|---|---|---|
| Guess the most common processes (ignores the photo) | 68% | 73% | 51% | 75% |
| VLM alone (its own top-8 ranking) | 77% | 84% | 55% | 88% |
| Current Hybrid (rules delete processes) | 14% | 28% | 7% | 14% |
| Original 92 rules as evidence + VLM | 77% | 83% | 55% | 88% |
| All 564 rules + VLM | 76% | 82% | 56% | 89% |
| 142 checked rules + VLM | 76% | 85% | 56% | 92% |
| 142 checked rules alone (no VLM) | 34% | 70% | 50% | 87% |

**Difference from the VLM alone, in percentage points** (95% range from resampling the photos; a range that includes 0 means no clear difference):

Test, 100 photos:

| Compared with the VLM alone | Top pick correct | Right family | All correct in top 3 | All correct in top 8 |
|---|---|---|---|---|
| All 564 rules + VLM | −1 (−3 to 0) | 0 (0 to 0) | +1 (−3 to +6) | 0 (−3 to +3) |
| 142 checked rules + VLM | −2 (−7 to +3) | −2 (−5 to 0) | −2 (−7 to +3) | +2 (−2 to +6) |

Dev, 535 photos:

| Compared with the VLM alone | Top pick correct | Right family | All correct in top 3 | All correct in top 8 |
|---|---|---|---|---|
| All 564 rules + VLM | −1 (−3 to 0) | −2 (−4 to −1) | +1 (−1 to +2) | +1 (0 to +2) |
| 142 checked rules + VLM | −1 (−3 to +1) | +1 (0 to +2) | +1 (−1 to +3) | +4 (+2 to +6) |

The same tables for the HARD photos (parts where injection molding is not the only answer) are in `outputs/soft_v3_summary_percent.json`; they tell the same story.

### What this says

1. **Rules that delete processes are the main problem.** The current Hybrid gets the top pick right on 19% of test photos, while guessing the most common processes gets 58%. The same 92 rules used as evidence on top of the VLM get 72%.
2. **More rules on the current 11 features do not move the numbers on new photos.** All 564 rules + VLM stay within about a point of the VLM alone on every measure (top pick 73% vs 74%; all correct in top 3 48% vs 47%), and every range includes 0. The 142 rules that passed the check on dev photos raised "all correct in top 8" by 4 points there (88% to 92%) but by only 2 on test, and the top pick dropped 2: choosing rules on dev photos overfits a little.
3. **The rules carry some information on their own.** With no VLM at all, the 142 checked rules get the right family on 73% of test photos (guessing: 68%) and all correct processes in the top 3 on 39% (guessing: 28%). They just know much less than the VLM.
4. **The limit is perception, not rule count** (section 4). That is why the next step is the visible-mark pass, not more rules on the same 11 questions.

## 4. Why: what the VLM actually perceives

Observability of the 92 v2 clauses: 28 rely on visible features, 24 on estimated sizes, **40 on features a photo cannot show** (wall thickness, draft angle, production volume, continuous fiber).

How the rule check is scored: take one photo where a rule should fire and one where it should not. If the rule gives stronger evidence to the right photo, it was right on that pair. **50% means a coin flip**; 100% means always right.

Perception checks on the 635 photos (archived reads):

- `continuous_fiber_required`: answered "yes" for **86% of injection-molded parts** (85% for everything else); 54% of all replies are "unknown". The feature carries no signal, and the v2 clause "IM forbidden if continuous fiber" kills the most common correct answer.
- `hollow` (closed hollow body): "yes" for **59% of parts that are not blow molded**; repeated questions about the same photo disagree on 47% of photos.
- `constant_cross_section`: "yes" for only **20% of photos where pultrusion is a correct answer** (composite tubes, leaf springs); `axisymmetric` for only 39% of tubes and vessels.
- Wall-thickness rules point the wrong way on real photos: right on only **37-39% of photo pairs**, worse than a coin flip.
- Surprise in the other direction: production volume cannot be seen, but the VLM's guess is useful. Prototype vs mass-production rules are right on **87% of photo pairs**. Which features count as "visible" is only a starting point; the check decides.

Rule check on dev (`outputs/rule_validity_v3.json`): of the 295 rules on current features (289 have readings on dev photos), **142 point the right way** (right on at least 55% of photo pairs and firing on at least 5 photos) and **28 point the wrong way** (right on under 45%). By group: G1 35 of 92, G2 65 of 118, G3 28 of 60, G5 14 of 25.

## 5. How to run

From the repository root (Python 3.9+, stdlib only except the perception pass):

```
python scripts/extract_evidence_v3.py          # archived reads -> outputs/evidence_v2pool.json (needs the extracted vlm_cache)
python scripts/build_rules_v3.py               # -> data/rules_v3.json
python scripts/rule_validity_v3.py             # -> outputs/rule_validity_v3.json
python scripts/eval_soft_v3.py                 # picks settings on dev (+ --test for Gold-100 with frozen settings)
python scripts/summary_percent_v3.py           # the percentage tables in section 3, with 95% ranges
```

Visible-signature pass (needs `pip install "mip[gemini]"`; reads GEMINI_API_KEY from the environment or the repo `.env`):

```
python scripts/perceive_signatures_v3.py --limit 60     # pilot: first 60 dev photos x 3 samples = 180 calls
python scripts/rule_validity_v3.py                      # do the visible-mark rules point the right way?
python scripts/perceive_signatures_v3.py                # full pass: 635 photos x 3 = 1,905 calls, resumable
```

The pass asks all 28 signature questions in one call per photo and returns JSON. It runs dev photos first and Gold-100 last, so a pilot never touches the test set. It is cached and resumable under `outputs/vlm_cache/sig-v3-gemini-2.5-flash-lite/`, and it checks each photo's sha against bench-v2 before calling.

## 6. Commitments for the signature evaluation (set before any signature data exists)

- Same photos, same split: dev = 535, test = Gold-100; the same four percentages as section 3.
- A visible-mark rule counts only if it passes the same check on dev photos (right on at least 55% of photo pairs, firing on at least 5 photos).
- Configs chosen on dev with the same grid, frozen with a file hash, test scored once.
- Proposed success criterion (confirm before the full pass): on the Gold-100 test photos, **"all correct in top 3" rises at least 5 percentage points above the VLM alone (47% today), with the whole 95% range above 0**. All four percentages get reported either way. If it fails, that is the finding: visible marks are not perceived reliably enough either, and the paper reports it.
- This criterion was restated in plain percentages on 2026-10-12, before any visible-mark data existed (the first draft used an averaged ranking score).

Frozen-artifact hashes for this run (sha256):
- `outputs/soft_v3_frozen_dev_configs.json` 8bbe58d12bc870d1942641192d6cfe6d164eccc6258b2f1c2f6365156cb4fb69
- `data/rules_v3.json` 13f7ef36c9475db4220dfb228a0be94b3e4237881664c591dcd3d802d43f496b
- `outputs/rule_validity_v3.json` 603c5347e01f4c8c7175a7d5937859422013d078f842931f02d530625a91eaaa

## 7. Caveats

- Labels on the real pool are category-level feasible sets, not per-photo expert annotations. Gold-100 per-photo labels remain the fix.
- Dev and test share part categories, so this measures generalization to new photos, not to new kinds of parts.
- v3 weights are set by hand from domain knowledge. Only four global settings (prior weight, support threshold, global scale, observability weights) were tuned, and only on dev.
- All v3 rules are drafts: citations by topic, `[VERIFY]`, no adversarial pass yet.

## 8. Files (all new)

| File | What |
|---|---|
| `data/rules_v3.json`, `data/rules_v3_review.csv` | the 564-rule base; the CSV shows, per rule, how often it points the right way on dev photos (in %), for review in Excel |
| `src/mip/evidence.py` | soft-evidence scorer with per-rule provenance |
| `scripts/build_rules_v3.py` | generates rules_v3.json (edit rules here) |
| `scripts/extract_evidence_v3.py` | archived reads -> evidence_v2pool.json |
| `scripts/rule_validity_v3.py` | rule and feature validity on dev |
| `scripts/eval_soft_v3.py`, `scripts/summary_percent_v3.py` | settings on dev; the percentage tables with 95% ranges (`bootstrap_v3.py` is now an alias) |
| `scripts/perceive_signatures_v3.py` | the visible-signature perception pass |
| `outputs/image_sha24_v2pool.json`, `evidence_v2pool.json` | photo-to-cache map; extracted reads |
| `outputs/soft_v3_summary_percent.json` | **the results in percentages** (section 3) |
| `outputs/soft_v3_dev.json`, `soft_v3_test.json`, `soft_v3_frozen_dev_configs.json`, `rule_validity_v3.json` | raw results and frozen settings (fractions, not percentages) |
| `outputs/soft_v3_bootstrap.json` | superseded: the earlier ranking-score version |

## 9. Addendum 2026-10-13: learned trust per feature (dev only, frozen; Gold-100 not yet scored)

**What.** Instead of hand-set weights, the dev photos set one trust number per perceived feature (the 11 features, plus one for rules that combine features). The hand-written rule weights and directions are kept (trust is never negative, so a rule can only act as written), and the VLM's own top-8 slots and direct answer are calibrated alongside. There are no per-process base rates, so the model cannot memorize part types: about 22 learned numbers in total. Script: `scripts/trust_v3.py` (fitting needs `pip install numpy scipy`; scoring with the frozen weights is standard library only). Visible-mark reads are not used.

**Dev results, % of photos; every photo is scored by a model that never saw it (5-fold cross-validation).** "Unseen part types" holds whole part types out of training.

| Model | Top pick correct | Right family | All correct in top 3 | All correct in top 8 |
|---|---|---|---|---|
| VLM alone (its own ranking) | 77.0% | 83.7% | 55.1% | 88.2% |
| Random folds: VLM signals only, learned, no rules | 77.4% | 84.5% | 55.5% | 89.0% |
| Random folds: trust per rule | 78.3% | 84.5% | 60.7% | 93.5% |
| Random folds: **trust per feature** | 77.6% | 84.5% | 57.2% | 91.2% |
| Unseen part types: VLM signals only, learned, no rules | 77.4% | 84.7% | 55.7% | 89.0% |
| Unseen part types: trust per rule | 76.4% | 84.1% | 51.0% | 84.7% |
| Unseen part types: **trust per feature** | 77.4% | 84.3% | 57.0% | 90.8% |

Gains of trust per feature, in percentage points (95% range from resampling photos):
- vs the VLM alone, random folds: top 8 +3.0 (+1.1 to +4.9); top 3 +2.1 (−0.2 to +4.3); top pick +0.6 (−1.5 to +2.6)
- vs the VLM alone, unseen part types: top 8 +2.6 (+0.9 to +4.5); top 3 +1.9 (−0.6 to +4.5); top pick +0.4 (−1.7 to +2.4)
- vs learned VLM signals without rules (the rules' own share): top 8 +2.2 (+0.6 to +4.1) random folds, +1.9 (+0.2 to +3.6) unseen part types

**Findings.**
1. Learned trust per feature beats the VLM alone on keeping every correct process in the top 8, including on part types never seen in training. Top 3 rises about 2 points (not yet significant); the top pick does not change.
2. A trust weight per rule looks best on familiar part types (top 3 +5.6, top 8 +5.2) but loses on unseen part types (top 3 −4.1, top 8 −3.6): it memorizes part types. With free signs and per-process base rates, 69 of the 202 larger weights pointed against their own rule (`scripts/explore/cv_ideas2.py`).
3. The learned trust profile (frozen model): production volume 0.53, undercut 0.29, feature combinations 0.24, rib count 0.15, hollow 0.13, continuous fiber 0.12, axisymmetric 0.09, constant cross-section 0.08, max dimension 0.07; draft angle, projected area and wall thickness 0.00. This matches the rule check in section 4.
4. Rule agreement as a flag for VLM errors is weak: AUROC 0.57, against 0.59 for the VLM agreeing with itself (`scripts/explore/cv_ideas.py`).

**Frozen.** `outputs/trust_v3_frozen.json` sha256 23b9c1d17b361a87c696596619826f4585f4d47b6cc6d6281924be2f283c42a9 (also in `trust_v3_frozen.sha256`): lambda 10, tau 0.5, fit on all 535 dev photos, against `rules_v3.json` 13f7ef36... and `evidence_v2pool.json` 1514b9b6.... Dev cross-validation tables: `outputs/trust_v3_dev_cv.json`. Exploratory scripts behind these numbers: `scripts/explore/cv_ideas*.py` (numpy, scipy, scikit-learn).

**Protocol.** Gold-100 is scored once with `python scripts/trust_v3.py --test`, planned together with the per-photo labels (`--labels`); the script refuses a second scoring. This model was developed after the first Gold-100 scoring in section 3, so it is reported as a second, pre-declared look at Gold-100. A model that adds visible-mark reads is a new model and gets frozen under a new name.
