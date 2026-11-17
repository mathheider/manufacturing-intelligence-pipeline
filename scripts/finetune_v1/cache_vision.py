# -*- coding: utf-8 -*-
"""Run the frozen SigLIP encoder once per photo; save outputs (fp16) to data/vision_cache.pt."""
import time
import torch
from PIL import Image
from ft_common import DATA, VIS_CACHE, Runner, load_labels

E, _, _ = load_labels()
R = Runner()
vm = R.model.model.vision_model.eval()
feats, t0 = {}, time.time()
with torch.no_grad():
    for n, pid in enumerate(sorted(E)):
        im = Image.open(DATA / "photos" / f"{pid}.jpg").convert("RGB")
        pv = R.proc.image_processor(images=[im], return_tensors="pt")["pixel_values"][:, 0]
        feats[pid] = vm(pixel_values=pv).last_hidden_state[0].to(torch.float16)
        if n % 50 == 0:
            print(n, pid, tuple(pv.shape), f"{time.time() - t0:.0f}s", flush=True)
torch.save(feats, VIS_CACHE)
print("saved", len(feats), f"{time.time() - t0:.0f}s")
