# -*- coding: utf-8 -*-
"""MIP fine-tune v1: SmolVLM-256M-Instruct + LoRA on the bench-v2 dev photos (CPU).

    python run.py zeroshot                 # prompting baseline: variant chosen on dev, then Gold-100 once
    python run.py cv --scheme lopo --fold k    # unseen part types (trust_v3 folds, seed 7)
    python run.py cv --scheme random --fold k  # new photos of seen part types (trust_v3 folds)
    python run.py final                    # train on all 535 dev photos, freeze adapter + sha256
    python run.py test                     # score Gold-100 ONCE with the frozen adapter
    python run.py report                   # tables + paired gains with 95% ranges
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

from ft_common import (OUT, RECIPE, PROMPTS, TRAIN_PROMPT, MEASURES, Runner, load_labels, load_vision, dev_ids,
                       test_ids, folds, prevalence, ranked, per_photo, vlm_alone, table, gains, sha256, MODEL_DIR)

ADAPTER = OUT / "ft_v1_adapter"
FROZEN = OUT / "ft_v1_frozen.json"
TEST_OUT = OUT / "ft_v1_test.json"
ZS_OUT = OUT / "zeroshot.json"


def log(*a):
    print(datetime.now().strftime("%H:%M:%S"), *a, flush=True)


try:                                    # return freed heap to the OS each step (variable-length batches fragment it)
    import ctypes
    _trim = ctypes.CDLL("libc.so.6").malloc_trim
except Exception:                       # noqa: BLE001
    _trim = None


def rss_gb():
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return float("nan")


def train(R, vis, E, ids):
    rng = random.Random(RECIPE["seed"])
    torch.manual_seed(RECIPE["seed"])
    n_train, n_all = R.add_lora()
    log(f"trainable {n_train / 1e6:.2f}M of {n_all / 1e6:.1f}M params; {len(ids)} training photos")
    params = [p for p in R.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=RECIPE["lr"], weight_decay=RECIPE["weight_decay"])
    B = RECIPE["batch"]
    spe = math.ceil(len(ids) / B)
    total = RECIPE["epochs"] * spe
    warm = max(1, round(RECIPE["warmup_frac"] * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    R.model.train()
    step, t0, hist = 0, time.time(), []
    for ep in range(RECIPE["epochs"]):
        order = list(ids)
        rng.shuffle(order)
        run = []
        for b in range(0, len(order), B):
            batch = order[b:b + B]
            targets = [R.target_ids(rng.sample(E[i]["expert"], len(E[i]["expert"]))) for i in batch]
            v = torch.stack([vis[i].float() for i in batch])
            loss = R.batch_loss(v, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            run.append(loss.item())
            del loss, v
            if _trim is not None:
                _trim(0)
            if step % 10 == 0 or step == total:
                el = time.time() - t0
                log(f"ep {ep + 1} step {step}/{total} loss {np.mean(run[-10:]):.3f}  {el / step:.1f}s/step  "
                    f"eta {el / step * (total - step) / 60:.0f} min  rss {rss_gb():.2f} GB")
        hist.append(round(float(np.mean(run)), 4))
        log(f"epoch {ep + 1} mean loss {hist[-1]}")
    R.model.eval()
    return hist


def score_ids(R, vis, ids):
    S, t0 = {}, time.time()
    for n, i in enumerate(ids):
        s, m = R.score(vis[i][None].float())
        S[i] = {"sum": s, "mean": m}
        if n % 25 == 0:
            log(f"scored {n}/{len(ids)}  {(time.time() - t0) / (n + 1):.2f}s/photo")
    return S


def cmd_zeroshot(a):
    E, fam, procs = load_labels()
    vis = load_vision()
    dev, test = dev_ids(E), test_ids(E)
    order = prevalence(E, dev, procs)
    R = Runner()
    R.model.eval()
    res, dev_scores = {}, {}
    for key in PROMPTS:
        R.set_prompt(key)
        dev_scores[key] = score_ids(R, vis, dev)
        for norm in ("sum", "mean"):
            pp = {i: per_photo(ranked(procs, dev_scores[key][i][norm], order), set(E[i]["expert"]), fam) for i in dev}
            res[f"{key}/{norm}"] = {"dev": table(pp, dev),
                                    "select": float(np.mean([np.mean([pp[i][c] for c in (0, 2, 3)]) for i in dev]))}
            log("zero-shot", key, norm, res[f"{key}/{norm}"])
    best = max(res, key=lambda k: res[k]["select"])
    key, norm = best.split("/")
    log("chosen on dev:", best)
    R.set_prompt(key)
    ts = score_ids(R, vis, test)
    json.dump({"_meta": {"what": "SmolVLM-256M-Instruct, no fine-tuning, prompting baseline",
                         "variants_dev": res, "chosen_on_dev": best,
                         "selection": "mean of top pick / all in top 3 / all in top 8 on dev (as trust_v3)",
                         "prompt": PROMPTS[key]},
               "dev": {i: dev_scores[key][i][norm] for i in dev},
               "test": {i: ts[i][norm] for i in test}}, open(ZS_OUT, "w"), indent=0)
    log("wrote", ZS_OUT)


def cmd_cv(a):
    E, fam, procs = load_labels()
    vis = load_vision()
    F = folds(E, a.scheme)
    test = F[a.fold]
    train_ids = [i for k, f in enumerate(F) if k != a.fold for i in f]
    out = OUT / f"cv_{a.scheme}_fold{a.fold}.json"
    if out.exists():
        log("exists", out)
        return 0
    log(f"{a.scheme} fold {a.fold}: train {len(train_ids)} / held out {len(test)} "
        f"({len({E[i]['category'] for i in test})} part types held out)")
    R = Runner(TRAIN_PROMPT)
    hist = train(R, vis, E, train_ids)
    S = score_ids(R, vis, test)
    json.dump({"_meta": {"scheme": a.scheme, "fold": a.fold, "n_train": len(train_ids), "n_test": len(test),
                         "loss_per_epoch": hist, "recipe": {k: v for k, v in RECIPE.items()}},
               "train_ids": train_ids, "scores": {i: S[i]["sum"] for i in test}}, open(out, "w"), indent=0)
    log("wrote", out)


def cmd_final(a):
    E, fam, procs = load_labels()
    vis = load_vision()
    if FROZEN.exists() and not a.refreeze:
        log("already frozen:", FROZEN)
        return 1
    dev = dev_ids(E)
    R = Runner(TRAIN_PROMPT)
    hist = train(R, vis, E, dev)
    del vis
    if _trim is not None:
        _trim(0)
    log(f"rss before save {rss_gb():.2f} GB")
    R.peft.save_pretrained(str(ADAPTER))
    w = ADAPTER / "adapter_model.safetensors"
    meta = {"model": "MIP fine-tune v1: SmolVLM-256M-Instruct + LoRA", "base": "HuggingFaceTB/SmolVLM-256M-Instruct",
            "base_weights_sha256": sha256(MODEL_DIR / "model.safetensors"),
            "frozen_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "fit_on": f"all {len(dev)} dev photos (Gold-100 not read)", "recipe": RECIPE, "prompt": PROMPTS[TRAIN_PROMPT],
            "loss_per_epoch": hist, "adapter_sha256": sha256(w),
            "labels_sha256": sha256(OUT.parent / "data" / "labels.json"),
            "protocol": "score Gold-100 once with `run.py test`; per-photo scores saved so they can be re-scored "
                        "against per-photo labels later without re-running the model"}
    FROZEN.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log("frozen", meta["adapter_sha256"])


def cmd_test(a):
    from peft import PeftModel
    E, fam, procs = load_labels()
    if TEST_OUT.exists():
        log("Gold-100 already scored for this model:", TEST_OUT, "- refusing a second look")
        return 1
    meta = json.loads(FROZEN.read_text(encoding="utf-8"))
    assert sha256(ADAPTER / "adapter_model.safetensors") == meta["adapter_sha256"], "adapter changed since freezing"
    vis = load_vision()
    R = Runner(TRAIN_PROMPT)
    R.peft = PeftModel.from_pretrained(R.model, str(ADAPTER))     # injects the LoRA layers into R.model in place
    R.model.eval()
    test = test_ids(E)
    S = score_ids(R, vis, test)
    json.dump({"_meta": {"adapter_sha256": meta["adapter_sha256"], "scored_utc":
                         datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "n": len(test),
                         "labels": "bench-v2 category-level expert sets (per-photo Gold-100 labels pending)"},
               "scores": {i: S[i]["sum"] for i in test}}, open(TEST_OUT, "w"), indent=0)
    log("wrote", TEST_OUT)


def cmd_report(a):
    E, fam, procs = load_labels()
    dev, test = dev_ids(E), test_ids(E)
    zs = json.loads(ZS_OUT.read_text(encoding="utf-8")) if ZS_OUT.exists() else None
    rep = {"_meta": {"measures": MEASURES, "units": "% of photos; gains in percentage points, 95% range from "
                     "resampling photos (4000 draws, seed 20261012)"}}
    names = {"FT": "SmolVLM-256M + LoRA (fine-tuned)", "ZS": "SmolVLM-256M prompting (same model, no fine-tuning)",
             "M0": "Gemini 2.5 Flash-Lite alone (its own top-8, bench-v2 archive)",
             "PREV": "Guess the most common processes (ignores the photo)"}
    rep["_meta"]["methods"] = names
    for scheme in ("lopo", "random"):
        paths = [OUT / f"cv_{scheme}_fold{k}.json" for k in range(5)]
        if not all(p.exists() for p in paths):
            continue
        res = {m: {} for m in names}
        for p in paths:
            d = json.loads(p.read_text(encoding="utf-8"))
            order = prevalence(E, d["train_ids"], procs)
            for i, s in d["scores"].items():
                T = set(E[i]["expert"])
                res["FT"][i] = per_photo(ranked(procs, s, order), T, fam)
                res["M0"][i] = per_photo(vlm_alone(E, i, order), T, fam)
                res["PREV"][i] = per_photo(order, T, fam)
                if zs:
                    res["ZS"][i] = per_photo(ranked(procs, zs["dev"][i], order), T, fam)
        ids = sorted(res["FT"])
        hard = [i for i in ids if E[i]["hard"]]
        rep[scheme] = {"n": len(ids), "percent": {m: table(r, ids) for m, r in res.items() if r},
                       "percent_hard": {m: table(r, hard) for m, r in res.items() if r}, "n_hard": len(hard),
                       "gains": {f"FT vs {b}": gains(res["FT"], res[b], ids) for b in ("M0", "ZS", "PREV") if res[b]}}
    if TEST_OUT.exists():
        d = json.loads(TEST_OUT.read_text(encoding="utf-8"))
        order = prevalence(E, dev, procs)
        res = {m: {} for m in names}
        for i in test:
            T = set(E[i]["expert"])
            res["FT"][i] = per_photo(ranked(procs, d["scores"][i], order), T, fam)
            res["M0"][i] = per_photo(vlm_alone(E, i, order), T, fam)
            res["PREV"][i] = per_photo(order, T, fam)
            if zs:
                res["ZS"][i] = per_photo(ranked(procs, zs["test"][i], order), T, fam)
        hard = [i for i in test if E[i]["hard"]]
        rep["test"] = {"n": len(test), "percent": {m: table(r, test) for m, r in res.items() if r},
                       "percent_hard": {m: table(r, hard) for m, r in res.items() if r}, "n_hard": len(hard),
                       "gains": {f"FT vs {b}": gains(res["FT"], res[b], test) for b in ("M0", "ZS", "PREV") if res[b]}}
    (OUT / "ft_v1_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    for k, v in rep.items():
        if k == "_meta":
            continue
        print(f"\n== {k} (n={v['n']}) ==")
        for m, t in v["percent"].items():
            print(f"  {names[m]:62s} " + "  ".join(f"{x:5.1f}" for x in t.values()))
        for g, gv in v["gains"].items():
            print(f"  {g}: " + "  ".join(f"{q['gain_points']:+.1f} ({q['range95'][0]:+.1f} to {q['range95'][1]:+.1f})" for q in gv.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["zeroshot", "cv", "final", "test", "report"])
    ap.add_argument("--scheme", choices=["lopo", "random"], default="lopo")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--refreeze", action="store_true")
    a = ap.parse_args()
    return {"zeroshot": cmd_zeroshot, "cv": cmd_cv, "final": cmd_final, "test": cmd_test, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main() or 0)
