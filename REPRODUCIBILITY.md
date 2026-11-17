# Reproducibility contract

The whole point of this rebuild is that the results are **reproducible on any
machine, at any time, byte-for-byte** — no API key, no network, no third-party
packages, no dependence on process-level randomness.

## One command

```bash
PYTHONPATH=src python -m mip.cli bench --seeds 40 --direct-topk auto
# or, after `pip install -e .`:
mip bench --seeds 40 --direct-topk auto
```

## Expected output (must match exactly; re-locked 2026-07-01 after the
## Phase-1 rule densification + adversarial clause review, 71 -> 92 clauses)

```
backend=mock  dataset=seed  instances=14  trials=40  seed=20260630

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

## Other logged commands (Phase 2/3 artifacts)

```bash
# 1000-instance synthetic run -> outputs/benchmark_1000.json
PYTHONPATH=src python -m mip.cli bench --size 1000 --seeds 3 --direct-topk auto --out outputs/benchmark_1000.json

# results charts (values read from outputs/*.json, annotated to 3 decimals)
python assets/make_charts.py

# real-image pool: fetch -> QC -> validate -> coverage -> sample sheet
python scripts/fetch_images.py --per-category 70
python scripts/qc_images.py --apply
python scripts/check_manifest.py
python scripts/coverage_report.py
python scripts/contact_sheet.py --n 30

# real-image benchmark, stratified 150-image subset on flash-lite (fits one
# 10k-requests/day quota window; own cache dir bench-gemini-2.5-flash-lite)
PYTHONPATH=src python -m mip.cli bench --dataset real --backend gemini --model gemini-2.5-flash-lite --stratified --limit 150 --direct-topk 8 --workers 6 --confirm --out outputs/benchmark_real.json

# FULL real-image pool on gemini-2.5-flash: needs ~36k calls total, i.e. 3-4
# daily 10k-request quota windows. The run is RESUMABLE: re-running the same
# command replays outputs/vlm_cache/bench hits for free and only pays for new
# images, so run this once per day until it completes:
PYTHONPATH=src python -m mip.cli bench --dataset real --backend gemini --stratified --direct-topk 8 --workers 6 --confirm --out outputs/benchmark_real_full.json

# rule-citation status (proposed [VERIFY] vs verified)
python scripts/citation_status.py
```

The `mip demo` (8 trials) and the `pytest` golden regression (5 trials) lock
smaller, faster versions of the same run.

## Why it is deterministic

1. **Seeds derive from SHA-256, never `hash()`.** Python salts the built-in
   `hash` per process (`PYTHONHASHSEED`), so it can never seed an experiment.
   `mip/_seed.py` hashes the string form of the seed components with SHA-256 —
   identical on every machine. Verify:

   ```bash
   PYTHONHASHSEED=0   python -m mip.cli demo | md5sum
   PYTHONHASHSEED=999 python -m mip.cli demo | md5sum   # same hash
   ```

2. **MockVLM noise is a pure function of `(seed, feature, sample_index)`.** Two
   VLMs built with the same seed return the same value for the same feature's
   same-index read, regardless of call order. So every pipeline in the ablation
   perceives *identical* noisy reads — the comparison isolates decision logic,
   not perception luck. (The original `demo.py` gave each method a different
   seed; this is the fairness fix.)

3. **The rule engine is a pure function.** `eliminate()` returns a new survivor
   set and never mutates inputs or global state.

4. **The bootstrap CI is seeded.** Confidence intervals are stable across runs
   (they do depend on `--n-boot`, which is why the golden regression checks
   point estimates, not intervals).

5. **Config is echoed into every output.** `outputs/benchmark_summary.json`
   embeds the full config + seed, so a result is never separable from what
   produced it.

## Environment used to lock the golden numbers

- Python 3.14 (any 3.9+ works; the numbers do not depend on the minor version)
- Third-party packages: **none** for the offline benchmark
- OS: Windows 11 (platform-independent — no OS-specific behavior)

## If a number moves

The `tests/test_benchmark_regression.py` golden values will fail. That is the
alarm: some change touched the pipeline, the seeding, the rules, or the dataset.
Re-derive intentionally (run the capture snippet in that test's docstring) and
update the golden constants **in the same commit** as the change that moved
them, with a note in `docs/IMPROVEMENTS.md` explaining why.
```bash
python -m pytest tests/test_benchmark_regression.py -q
```
