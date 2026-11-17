# GOLD-100 — Curated Evaluation Set Protocol

**Date declared:** 2026-09-01 (before any gold-set scoring; same discipline as PREREGISTRATION_BENCH_V2.md)
**Decision:** scale the headline evaluation down to a 100-image curated set ("Gold-100"), with curated non-open-source images first (my advisor's priority), while the full open pool remains the large-n appendix and the public release. All 17 processes stay covered.

## 1. Composition rule (fixed before scoring)

0. **Curated non-open-source images join first.** The 14 seed archetypes in `data/components.json` reference image files (`steering_wheel.png`, …) that are **not present in the repo as of 2026-09-01**. When the files are supplied (place them in `images_curated/`), they enter the set first and displace the lowest-ranked *unprotected* pool picks. They carry full per-instance expert labels and the feature oracle — the only images for which the RuleSet ceiling is defined on photos.
1. Eligible pool photos = QC-accepted (635). **Rank = min(width, height), descending** (image-intrinsic; QC has no numeric score), tie-break by path.
2. Coverage pass: processes in ascending support order each take their 5 best-ranked photos (dedup) — rare processes protected first.
3. Fill to 100 by global rank.

Selection uses only image-intrinsic criteria and label coverage. **Model performance is never a selection input.** Manifest + audit: `images_pool/gold100_manifest.json`.

## 2. Resulting set (pool-only interim, until curated files arrive)

n=100 · 35 coverage-protected · min resolution 720 px · every process ≥5 (IM 58, SMC 27, CM 21, RTM 20, Overmolding 16, BMC 14, FFF/SLS/SLA 9, Hand lay-up 7, FW/Pultrusion/VARTM 6, SRIM/Thermoforming/Blow/LOM 5) · 71/100 multi-label · IM in 58/100.

## 3. Purpose and limits

- **Purpose:** the set that gets **per-instance expert labels** (target: validation by my advisor's group — 100 images is a feasible labeling pass; 635 was not). Until relabeled, pool images keep category-level feasible sets, and every gold-set figure carries that caveat.
- **Supports:** headline metrics with floors, family-level results, qualitative per-image analysis, expert-label comparison once labeled.
- **Does not support:** per-process case studies or stable confusion matrices (~6 images per rare process; 95% CI on a containment near .3 is roughly ±9 pts at n=100). Those stay on the large pool (592 after the n≥10-category trim; full 635 in the release).

## 4. Exploratory readout (records replay, 2026-09-01 — NOT confirmatory)

Same qualitative picture as the full pool, so the subset does not distort: Direct top-1 family accuracy .840 ALL / .816 HARD vs floors .680/.579; Direct setF1 .540/.422 vs const-IM floor .443/.268; Hybrid .163/.186 with containment .130/.092; Direct@8 containment .880/.842. (n=100/76; all from `records_real_v2.jsonl`, zero new VLM calls.)

## 5. Scoring plan

Pool members of the set score from the archived bench-v2 cache (free). Curated images have no cache — scoring them is a small new VLM run (~14 images), to be declared as bench-gold-v1 with its own cache archive before it runs. Confirmatory gold-set claims wait for the expert label pass; everything before that is labeled exploratory.
