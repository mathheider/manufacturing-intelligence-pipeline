# Preregistration: Full-Pool Real-Image Confirmatory Run (bench-v2)

Author: Mathias Heider. Committed BEFORE any bench-v2 full-pool inference.
Status: FROZEN 2026-08-04. H3 threshold chosen by Mathias (.20) and committed before
any bench-v2 full-pool inference; no edits after this point except dated deviation logs.

## Motivation
The 150-photo v1 run (outputs/benchmark_real.json, Gemini 2.5 Flash-Lite) showed the
Hybrid collapsing to macro-F1 .086 on HARD while per-feature provenance localized 84%
of false eliminations to two boolean reads; the 'hollow' question phrasing alone caused
58% (open shells photographed head-on read as hollow bodies). The phrasing was fixed
(bench-v2). This document freezes the confirmatory design before results are seen.

## Hypotheses (stated before the run)
- H1: bench-v2 phrasing reduces false eliminations attributable to the hollow read by
  at least 50% relative to v1 (v1 baseline: 58% of 281 false eliminations).
- H2: Hybrid macro-F1 on the full pool (HARD subset) improves over the v1 value of .086.
- H3 (falsification line): if Hybrid HARD macro-F1 on the full pool remains below .20,
  then perception calibration, not the architecture, remains the binding constraint,
  and the dissertation milestone ordering (calibrated perception first) stands as
  stated in the July deck. Threshold committed 2026-08-04, before the run. Anchors:
  v1 real-photo Hybrid .086; clean-feature Hybrid .385 HARD (CI [.375, .395]);
  clean-feature Direct .136 HARD. .20 sits above double the v1 collapse and above
  clean-feature Direct, but below half the clean-feature Hybrid ceiling: crossing it
  requires genuine perception repair, not noise.

## Frozen design
- Pool: all 41 part categories, every one of the 17 processes covered by >= 2
  categories; target per-category counts per the fetch plan; CC-licensed photos only,
  deduplicated; QC v2 (domain-locked prompt) acceptance rules unchanged from the July
  runs. Pool is closed BEFORE scoring begins; no photo added or removed after.
- Backend: Gemini 2.5 Flash-Lite, same prompt scaffolding as v1 except the
  preregistered bench-v2 phrasing changes; every raw response archived.
- Seed lineage: 20260630; same consistency-sampling config as v1
  (n_samples 5, tol_rel 0.1, min_agree 0.6, max_reprobe 1).
- Metrics: identical to v1 and to benchmark_1000.json: set-valued macro-F1, precision,
  recall, containment, shortlist F1@3, false eliminations per part, avg probes;
  HARD = expert set beyond injection molding; seeded-bootstrap 95% CIs.
- Analysis: the same scoring script as v1, run once, on the closed pool. Provenance
  tallies regenerated from THIS run (not the diag run).

## Commitments
- Results ship in deck s26 (before/after arc) and report section 8.6 whether they
  confirm or refute H1-H3.
- No metric, subset, threshold, or pool change after first sight of results.
- Direct, Direct@8, Greedy, Rule Set comparisons all reported, as in v1.
- Deviations, if forced (e.g., API deprecation), are logged in this file with dates,
  before proceeding.

Date committed: 2026-08-04  Repo commit hash: c84fbc3 (the freezing commit; this
line was filled by the immediate follow-up commit, since a commit cannot contain
its own hash).

## Execution log (dated; append-only)

- 2026-08-04: POOL CLOSED per the frozen design, before any bench-v2 inference.
  QC v2 screen completed over the full fetched pool (4,324 manifest entries):
  635 images accepted across all 41 categories; every one of the 17 processes
  covered by >= 2 categories; rejects moved to images_pool/_rejected with
  reasons archived in outputs/qc_report.json. Sample sheet eyeballed by Mathias
  (30-image contact sheet) and close-at-635 approved over a further replenish
  cycle. images_manifest.json SHA-256 at close:
  F25916B1AE6031A19421592A99309AFBF6AF5C45A224C3DA00299FD8D75ADF53.
  No photo is added or removed after this point.
- 2026-08-04: bench-v2 full-pool run started (gemini-2.5-flash, resumable
  daily until complete; PROMPT_VERSION bench-v2 confirmed active in
  src/mip/vlm/cache.py; every raw response archived in the run cache).
- 2026-08-05: DEVIATION, disclosed interim readout, logged BEFORE the analysis
  was run. At the candidate's direction for defense preparation, the photos
  whose complete response sets are already archived in the run cache are scored
  ONCE with the unmodified pipeline code (scripts/score_v2_interim.py:
  cache-only VLM, any cache miss skips the photo; no phrasing, metric,
  threshold, or pool change of any kind). The stratified launch order makes
  this cached prefix category-balanced by construction (see
  dataset.load_real). Output archived as outputs/benchmark_real_v2_interim.json
  and labeled INTERIM wherever cited (deck s27, report 8.6). H1-H3 remain
  judged SOLELY on the completed closed pool: the interim number never
  replaces benchmark_real_v2.json and ships alongside the final result
  regardless of which of the two is better.
- 2026-08-05: FORCED DEVIATION, logged before any further inference. The
  interim replay attempt exposed a concurrency bug in the runner as launched:
  with --workers 6, the shared cache wrapper's channel prefix and sample
  index raced across threads, mis-keying the archived responses (feature
  answers filed under direct-/topk- channels, sample indexes over-running
  their caps). Live pipeline computation received correct responses, but the
  archive was unreplayable: a resume would have replayed answers to the wrong
  questions and silently corrupted the confirmatory result. Remedy, applied
  before any result existed: (a) the entire first-window cache quarantined,
  not deleted, to outputs/vlm_cache/bench_raced_20260805/ (every raw response
  self-describes its true question in its JSON body, so the record survives);
  (b) src/mip/benchmark.py patched to build one cache wrapper per instance
  (thread-safe by construction; inner client shared); (c) the run restarts
  clean next quota window with the identical frozen design. No metric,
  phrasing, pool, or threshold changed. The 2026-08-05 interim-readout
  deviation above is deferred until the restarted run's correctly keyed
  cache holds a scoreable stratified prefix; the n=1 artifact produced before
  the bug was understood was quarantined alongside the raced cache.
- 2026-08-05: CLEAN RESTART LAUNCHED (container execution). The restart runs
  from the project's cloud (Cowork) container rather than the candidate's
  machine: the overnight device run stalled (machine sleep), and the desktop
  bridge cannot type into terminals, so container execution is the only
  supervisable path before the defense. Also corrected at launch: the
  2026-08-04 start entry recorded gemini-2.5-flash; the frozen design (v1
  parity) specifies gemini-2.5-flash-lite, and the restart passes
  --model gemini-2.5-flash-lite explicitly (cache dir
  outputs/vlm_cache/bench-gemini-2.5-flash-lite, fresh; the quarantined raced
  cache is not readable by this run, and v1's bench-v1 keys cannot collide
  because PROMPT_VERSION differs). The raced 2026-08-04 window produced no
  read results, so this correction changes no number. Run inputs
  hash-verified against the repository before launch: pool = the same closed
  635 (every photo md5-verified against the device copy; stratified
  interleave confirmed: the first-38 prefix covers all 38 categories);
  rules.json = committed HEAD 1201ed4 content
  (md5 3556e988d43b81a2a31afcf7cb141c8b), verified structurally identical to
  the frozen c84fbc3 table (same 17 processes, same 11 feature definitions
  including every VLM question prompt, same 92 forbidden_if
  (feature, op, value) triples; the diff is citation text and verification
  metadata only — CC-2's source closure); benchmark.py = the 6da498c
  thread-safety patch (md5 bed9f86e7338f7c12f549fb8ff648f6f; mock goldens
  reproduced exactly before launch). Command: PYTHONPATH=src python -m
  mip.cli bench --dataset real --backend gemini --model
  gemini-2.5-flash-lite --stratified --direct-topk 8 --workers 6 --confirm
  --out outputs/benchmark_real_v2.json (config seed 20260630, n_samples 5,
  all defaults unchanged; CLI estimate ceiling 43,180 calls). On completion
  (or at the deferred interim readout), the response cache and
  benchmark_real_v2.json transfer back to this repository and are committed;
  H1-H3 are judged solely on the completed 635-photo pool, and results ship
  regardless of outcome.
- 2026-08-05 (05:31 UTC): INFRASTRUCTURE RESTART, design unchanged. The 04:46
  container process was killed silently at ~05:02 UTC (no Python traceback, no
  in-process error in the log; consistent with an out-of-memory or sandbox
  kill). 6,189 archived responses survived intact and replay byte-identically.
  Remedy, infrastructure only: (a) GeminiVLM's decoded-image memoization
  removed -- the unbounded cache held every distinct photo ever loaded, growing
  without bound across a 635-photo run; the request bytes per call are
  identical and prompts/model/sampling are untouched; (b) the run now executes
  under a supervisor loop that relaunches on abnormal exit and resumes from
  the archive, exactly the resumability the cache design exists for;
  (c) scripts/score_v2_interim.py's cache path corrected to the per-model
  directory the corrected launch actually writes
  (outputs/vlm_cache/bench-gemini-2.5-flash-lite), matching benchmark.py's
  per-model cache naming -- a path fix, not a scoring change. Monitoring
  cadence tightened; every window's calls land in the same archive, so
  repeated kills cost time, never data.
- 2026-08-05 (07:05:57 UTC): RUN COMPLETE. The confirmatory bench-v2 run
  finished on the closed 635-photo pool, gemini-2.5-flash-lite, prompt version
  bench-v2, seed 20260630, all defaults as frozen. Execution detail for the
  record: after two detached container windows were silently killed (~05:02
  and ~05:44 UTC), the remainder ran as nine supervised foreground slices;
  the archive absorbed every window (41,420 raw responses; final slice
  reported 40,057 hits / 1,363 paid calls; a full cache-only replay scored
  635/635 photos with zero misses, confirming the archive is complete and
  deterministic). Verdicts against the committed bars, reported exactly:
  H1 NOT MET: hollow-attributed false eliminations fell 1.087 -> 0.557 per
  part, a 48.7% reduction against the >=50% bar (hollow's share of all false
  eliminations fell 58% -> 38.4%; continuous_fiber_required is now the top
  remaining contributor at 35.0%). H2 CONFIRMED: Hybrid HARD macro-F1
  0.198, 95% CI [0.176, 0.223], vs the v1 value 0.086 (2.3x; CI floor above
  the v1 point). H3 NOT CROSSED: 0.198 < 0.20 (the interval straddles the
  line; the committed rule reads the point estimate), so the calibration-first
  milestone ordering stands as preregistered. Artifacts committed with this
  entry: outputs/benchmark_real_v2.json (the confirmatory result),
  outputs/h1_tally_v2.json + scripts/h1_tally.py (the H1 provenance tally,
  cache-only replay), outputs/benchmark_real_v2_interim.json (the disclosed
  mid-run readout, n=108, Hybrid HARD 0.161 -- superseded by and consistent
  with the final number), outputs/assets/v2_before_after.png (the stamped
  before/after chart), and the full response archive
  (extracted at outputs/vlm_cache/bench-gemini-2.5-flash-lite/, which repo
  convention git-ignores; committed in tracked form as
  outputs/vlm_cache_archives/v2_cache_20260805.tar.gz). Deck s27 and report 8.6
  carry the final numbers; the interim figure is retained only as the labeled
  artifact. Results ship as committed, near-misses unrounded.
