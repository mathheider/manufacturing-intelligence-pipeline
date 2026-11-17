# Fine-tune v1: SmolVLM-256M + LoRA on the bench-v2 photos

Code: `scripts/finetune_v1/` (`ft_common.py`, `cache_vision.py`, `run.py`). Results: `outputs/finetune_v1/`.

## What was trained

- **Base model:** `HuggingFaceTB/SmolVLM-256M-Instruct` (Idefics3 architecture: SigLIP vision encoder 86M, connector 7M, SmolLM2 language model 135M). Open weights, small enough to train on CPU.
- **Method:** LoRA (r 16, alpha 32, dropout 0.05) on every attention and MLP projection of the language model and on the vision-to-text connector. The SigLIP encoder is frozen; its outputs are computed once per photo (one 512x512 tile, 64 image tokens).
- **Target:** given one photo and the question "Which manufacturing process was used to make this part?", answer with the expert's correct processes, separated by semicolons. The order is reshuffled every epoch, so the model learns to put probability on every correct process, not on a fixed first one.
- **Ranking at test time:** the log-probability the model gives each of the 17 process names as its answer. Sorting by it gives a full ranking, scored with the same four percentages as rules v3 and trust v3.
- **Fixed recipe, chosen before any result:** AdamW, learning rate 2e-4, 3 epochs, batch 8, 5% warmup then cosine decay, gradient clipping 1.0, seed 0. Nothing was tuned on any held-out photo.

## Data

- The 635 bench-v2 photos (hash-checked against `outputs/image_sha24_v2pool.json`), labels from `outputs/evidence_v2pool.json`: dev = 535 photos, test = Gold-100.
- Labels are the bench-v2 category-level expert feasible-process sets (17 processes, about 2 correct per part). Per-photo Gold-100 labels are still pending; per-photo scores are saved so the test can be re-scored against them without re-running the model.

## Protocol

- **Unseen part types:** 5-fold leave-part-type-out on dev, the same folds as `scripts/trust_v3.py` (seed 7). Every dev photo is scored by a model that never saw its part type.
- **Gold-100:** the model trained on all 535 dev photos is frozen (adapter sha256 in `ft_v1_frozen.json`) and scores Gold-100 once (`run.py test` refuses a second look).
- **Baselines on the same photos:** Gemini 2.5 Flash-Lite alone (its own top-8 from the bench-v2 archive), the same SmolVLM prompted without fine-tuning (prompt and length normalization chosen on dev), and the no-look prevalence guess.

## Results (% of photos; `outputs/finetune_v1/ft_v1_report.json`)

**Test: Gold-100, held out, scored once (100 photos; new photos of part types seen in training)**

| Method | Top pick correct | Right family | All correct in top 3 | All correct in top 8 |
|---|---|---|---|---|
| Guess the most common processes (ignores the photo) | 58% | 68% | 28% | 56% |
| Gemini 2.5 Flash-Lite alone (prompted, its own top-8) | 74% | 83% | 47% | 88% |
| SmolVLM-256M prompted (no fine-tuning) | 59% | 68% | 29% | 40% |
| **SmolVLM-256M + LoRA (fine-tuned)** | **56%** | **68%** | **43%** | **73%** |

Fine-tuned minus prompted (same model), percentage points with 95% range: top pick −3 (−8 to +2), right family 0 (−4 to +4), **all correct in top 3 +14 (+5 to +23)**, **all correct in top 8 +33 (+22 to +44)**.
Fine-tuned minus Gemini alone: −18 (−27 to −10), −15 (−22 to −9), −4 (−13 to +4), −15 (−24 to −7).

**Dev: unseen part types (5-fold leave-part-type-out, 535 photos)**

| Method | Top pick correct | Right family | All correct in top 3 | All correct in top 8 |
|---|---|---|---|---|
| Guess the most common processes | 67.7% | 72.5% | 44.9% | 67.7% |
| Gemini 2.5 Flash-Lite alone | 77.0% | 83.7% | 55.1% | 88.2% |
| SmolVLM-256M prompted | 67.1% | 72.3% | 44.9% | 52.9% |
| **SmolVLM-256M + LoRA** | **66.2%** | **71.8%** | **46.0%** | **66.7%** |

Fine-tuned minus prompted: −0.9, −0.6, +1.1 (−0.7 to +3.0), +13.8 (+9.9 to +17.8). Fine-tuned minus guessing: −1.5, −0.7, +1.1, −0.9.

HARD photos (injection molding is not the only answer) are in the report JSON. They tell the same story; on Gold-100 HARD, all correct in top 3 goes from 7% (prompted) to 25% (fine-tuned), with Gemini at 30%.

### What this says

1. **Fine-tuning teaches the small model which processes belong together.** On held-out photos it puts every correct process in its top 3 on 43% of photos (29% prompted) and in its top 8 on 73% (40% prompted).
2. **It does not learn to see manufacturing.** Its top pick is injection molding on 88 of 100 test photos, the same prior collapse the paper documents for Gemini, and it does not beat guessing. On part types it never saw, it is at the guessing level on every measure.
3. **A small open model fine-tuned on 535 photos stays well below a large prompted model** (Gemini 2.5 Flash-Lite) on every measure except top 3, where the gap is not significant (−4, range −13 to +4).
4. This supports the paper's position: the limit is perception. With category-level labels, the training signal is "which part type is this", not "which marks show the process". The fix is better perception targets (per-photo labels, visible-mark questions), not more fitting on the same labels.

### Caveats

- Gold-100 labels are category-level; per-photo labels will change the numbers. The saved per-photo scores (`ft_v1_test.json`) can be re-scored against them without re-running the model.
- One small model, one fixed recipe, one seed, CPU only. A larger open VLM (Qwen2.5-VL-3B/7B, PaliGemma 2) on a GPU, with the vision encoder unfrozen, is the obvious next test.
- The prompting baseline for SmolVLM uses the short prompt and summed log-probabilities, chosen on dev; the listed-choices prompt was far worse for this model.

### Reproduce

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "transformers>=4.50,<5" peft pillow num2words huggingface_hub
# working dir: data/labels.json + data/photos/*.jpg (from outputs/finetune_v1/_transfer/), base_model/ = snapshot of HuggingFaceTB/SmolVLM-256M-Instruct
python cache_vision.py
python run.py zeroshot
for k in 0 1 2 3 4; do python run.py cv --scheme lopo --fold $k; done
MALLOC_ARENA_MAX=2 python run.py final      # frozen adapter sha256 abc8f3df4ed5df0575c448b17f549dc1741ac2c0aab6c487b0f86eb51c978e5c
python run.py test                          # refuses a second Gold-100 look
python run.py report
```

Runtime on 2 CPU cores: about 15 min for the vision cache, 15 min per training run, under 3 hours end to end.
