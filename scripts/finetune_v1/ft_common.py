# -*- coding: utf-8 -*-
"""Shared pieces for the MIP fine-tune v1 (SmolVLM-256M-Instruct + LoRA), CPU only.

Task: from one real photo, rank the 17 manufacturing processes; labels are the
bench-v2 expert feasible-process sets. Scored with the same four percentages as
rules v3 / trust v3 (top pick correct, right family, all correct in top 3 / top 8).

The SigLIP vision encoder stays frozen; its outputs are computed once per photo
(cache_vision.py). LoRA adapts the language model and the vision-to-text connector.
Ranking: log-probability the model gives each process name as its answer
(17 candidates, one prefix pass per photo with a shared KV cache).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.environ.get("MIP_FT_ROOT", Path(__file__).resolve().parent))   # working dir with data/ and outputs/
DATA = ROOT / "data"
OUT = ROOT / "outputs"
MODEL_DIR = Path(os.environ.get("MIP_FT_MODEL_DIR", ROOT / "base_model"))     # local snapshot of HuggingFaceTB/SmolVLM-256M-Instruct
VIS_CACHE = DATA / "vision_cache.pt"

NAMES = {
    "Blow_Molding": "blow molding",
    "Bulk_Molding_Compound": "bulk molding compound (BMC)",
    "Compression_Molding": "compression molding",
    "Filament_Winding": "filament winding",
    "Fused_Filament_Fabrication": "fused filament fabrication (FFF 3D printing)",
    "Hand_Lay_up": "hand lay-up",
    "Injection_Molding": "injection molding",
    "Laminated_Object_Manufacturing": "laminated object manufacturing",
    "Overmolding": "overmolding",
    "Pultrusion": "pultrusion",
    "Resin_Transfer_Molding": "resin transfer molding (RTM)",
    "Selective_Laser_Sintering": "selective laser sintering (SLS)",
    "Sheet_Molding_Compound": "sheet molding compound (SMC)",
    "Stereolithography": "stereolithography (SLA)",
    "Structural_Reaction_Injection_Molding": "structural reaction injection molding (SRIM)",
    "Thermoforming": "thermoforming",
    "Vacuum_Assisted_RTM": "vacuum-assisted RTM (VARTM)",
}
PROMPTS = {
    "short": "Which manufacturing process was used to make this part?",
    "listed": ("Which manufacturing process was used to make this part? Choose from: "
               + ", ".join(sorted(NAMES.values())) + "."),
}
TRAIN_PROMPT = "short"          # fixed before any result; the prompting baseline may use either (chosen on dev)

# fixed recipe, set before any result and never tuned on Gold-100
RECIPE = dict(lora_r=16, lora_alpha=32, lora_dropout=0.05, lr=2e-4, epochs=3, batch=8, warmup_frac=0.05,
              weight_decay=0.0, seed=0,
              targets=r".*text_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
                      r"|.*connector\.modality_projection\.proj$")
SEED_PHOTOS, SEED_TYPES, SEED_BOOT = 20261013, 7, 20261012     # same seeds as scripts/trust_v3.py
MEASURES = ("top pick correct", "right family", "all correct in top 3", "all correct in top 8")


def sha256(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_labels():
    L = json.loads((DATA / "labels.json").read_text(encoding="utf-8"))
    E, fam = L["photos"], L["family_map"]
    procs = sorted({p for v in E.values() for p in v["top8"]} | set(fam))
    assert procs == sorted(NAMES), procs
    return E, fam, procs


def dev_ids(E):
    return sorted(i for i, v in E.items() if v["split"] == "dev")


def test_ids(E):
    return sorted(i for i, v in E.items() if v["split"] == "test")


def folds(E, scheme):
    dev = dev_ids(E)
    if scheme == "random":
        ids = list(dev)
        random.Random(SEED_PHOTOS).shuffle(ids)
        return [ids[k::5] for k in range(5)]
    cats = sorted({E[i]["category"] for i in dev})
    random.Random(SEED_TYPES).shuffle(cats)
    return [[i for i in dev if E[i]["category"] in set(cats[k::5])] for k in range(5)]


def prevalence(E, ids, procs):
    c = Counter(p for i in ids for p in E[i]["expert"])
    return sorted(procs, key=lambda p: (-c[p], p))


def ranked(procs, score, order):
    pos = {p: k for k, p in enumerate(order)}
    return sorted(procs, key=lambda p: (-round(score[p], 9), pos[p]))


def per_photo(R, truth, fam):
    pos = {p: k for k, p in enumerate(R)}
    need = max(pos.get(t, 99) for t in truth)
    return (float(R[0] in truth), float(fam[R[0]] in {fam[t] for t in truth}), float(need < 3), float(need < 8))


def vlm_alone(E, i, order):
    return E[i]["top8"] + [p for p in order if p not in E[i]["top8"]]


def table(res, ids):
    return {MEASURES[c]: round(100 * float(np.mean([res[i][c] for i in ids])), 1) for c in range(4)}


def gains(a, b, ids):
    rng = np.random.default_rng(SEED_BOOT)
    d = np.array([np.subtract(a[i], b[i]) for i in ids])
    bs = np.sort(np.array([d[rng.integers(0, len(ids), len(ids))].mean(axis=0) for _ in range(4000)]), axis=0)
    return {MEASURES[c]: {"gain_points": round(100 * float(d.mean(axis=0)[c]), 1),
                          "range95": [round(100 * float(bs[100, c]), 1), round(100 * float(bs[3899, c]), 1)]}
            for c in range(4)}


# ----------------------------------------------------------------------------- model
class Runner:
    def __init__(self, prompt_key=TRAIN_PROMPT):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        torch.set_num_threads(2)
        self.proc = AutoProcessor.from_pretrained(MODEL_DIR, do_image_splitting=False)
        self.model = AutoModelForImageTextToText.from_pretrained(MODEL_DIR, dtype=torch.float32)
        self.tok = self.proc.tokenizer
        self.img_tok = self.model.config.image_token_id
        self.eou = self.tok.convert_tokens_to_ids("<end_of_utterance>")
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.eou
        self.set_prompt(prompt_key)
        self.cand = {p: self.tok(" " + n, add_special_tokens=False)["input_ids"] for p, n in NAMES.items()}
        self.sep = self.tok(";", add_special_tokens=False)["input_ids"]

    def set_prompt(self, key):
        from PIL import Image
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPTS[key]}]}]
        txt = self.proc.apply_chat_template(msgs, add_generation_prompt=True)
        enc = self.proc(text=txt, images=[Image.new("RGB", (512, 512))], return_tensors="pt")
        self.prefix = enc["input_ids"][0].tolist()
        assert sum(t == self.img_tok for t in self.prefix) == 64
        self.prompt_key = key

    # parts of the (possibly LoRA-wrapped, in place) model
    @property
    def text(self):
        return self.model.model.text_model

    def connect(self, vis):
        return self.model.model.connector(vis)

    def embed(self, ids, feats):
        emb = self.text.get_input_embeddings()(ids)
        mask = (ids == self.img_tok).unsqueeze(-1)
        return emb.masked_scatter(mask, feats.to(emb.dtype))

    def add_lora(self):
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=RECIPE["lora_r"], lora_alpha=RECIPE["lora_alpha"], lora_dropout=RECIPE["lora_dropout"],
                         target_modules=RECIPE["targets"], bias="none")
        self.peft = get_peft_model(self.model, cfg)
        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in self.model.parameters())
        return n_train, n_all

    def target_ids(self, procs_in_order):
        out = []
        for k, p in enumerate(procs_in_order):
            if k:
                out += self.sep
            out += self.cand[p]
        return out + [self.eou]

    def batch_loss(self, vis, targets):
        """vis [B,1024,768]; targets: list of token lists. Mean CE over answer tokens."""
        B = len(targets)
        L = len(self.prefix) + max(len(t) for t in targets)
        ids = torch.full((B, L), self.pad, dtype=torch.long)
        am = torch.zeros((B, L), dtype=torch.long)
        lab = torch.full((B, L), -100, dtype=torch.long)
        P = len(self.prefix)
        for b, t in enumerate(targets):
            seq = self.prefix + t
            ids[b, :len(seq)] = torch.tensor(seq)
            am[b, :len(seq)] = 1
            lab[b, P:P + len(t)] = torch.tensor(t)
        emb = self.embed(ids, self.connect(vis))
        h = self.text(inputs_embeds=emb, attention_mask=am).last_hidden_state
        sel = lab[:, 1:] != -100                           # positions whose next token is an answer token
        logits = self.model.lm_head(h[:, :-1][sel])
        return torch.nn.functional.cross_entropy(logits, lab[:, 1:][sel])

    @torch.no_grad()
    def score(self, vis):
        """vis [1,1024,768] -> {process: log p(name as the answer)}."""
        ids = torch.tensor([self.prefix])
        out = self.text(inputs_embeds=self.embed(ids, self.connect(vis)), use_cache=True)
        lp0 = torch.log_softmax(self.model.lm_head(out.last_hidden_state[:, -1]).float(), -1)[0]
        procs = sorted(self.cand)
        C = [self.cand[p] for p in procs]
        Lm = max(len(c) for c in C)
        cid = torch.full((len(C), Lm), self.pad, dtype=torch.long)
        for k, c in enumerate(C):
            cid[k, :len(c)] = torch.tensor(c)
        cache = out.past_key_values
        cache.batch_repeat_interleave(len(C))
        h = self.text(input_ids=cid, past_key_values=cache, use_cache=False).last_hidden_state
        lp = torch.log_softmax(self.model.lm_head(h).float(), -1)
        s, m = {}, {}
        for k, (p, c) in enumerate(zip(procs, C)):
            tot = lp0[c[0]].item() + sum(lp[k, j - 1, c[j]].item() for j in range(1, len(c)))
            s[p], m[p] = tot, tot / len(c)
        return s, m


def load_vision():
    return torch.load(VIS_CACHE)
