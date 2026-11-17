"""Offline VLM simulator.

Two behaviours make the pipeline's contribution observable without a real model:

1. **Noisy perception.** Each feature read is the ground-truth value perturbed by
   a type-appropriate noise model (gaussian on numerics, Bernoulli flip on
   booleans, jitter on integers). This is what makes single-pass perception
   unreliable and gives consistency sampling something to do.

2. **Corpus-frequency bias.** ``ask_process`` returns ``Injection_Molding`` with
   probability ``inj_molding_prior`` (default 0.90) regardless of the part --
   simulating the 217/217 injection-molding default observed in the preliminary
   data. This is the empirical failure mode the architecture is designed to
   neutralise.

Determinism: every draw is a pure function of ``(seed, feature_name,
sample_index)``. Two MockVLMs built with the same seed return the *same* value
for the same feature's same-index read, no matter what order or how many other
features were queried in between. That is what makes the multi-pipeline
comparison a controlled experiment rather than a noise race.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional

from .._seed import rng
from ..config import NoiseConfig
from .base import BaseVLM, VLMResponse

PROCESS_CHOICES = [
    "Injection_Molding", "Thermoforming", "Compression_Molding",
    "Resin_Transfer_Molding", "Vacuum_Assisted_RTM",
    "Structural_Reaction_Injection_Molding", "Sheet_Molding_Compound",
    "Bulk_Molding_Compound", "Hand_Lay_up", "Blow_Molding",
    "Pultrusion", "Filament_Winding", "Overmolding",
    "Fused_Filament_Fabrication", "Stereolithography",
    "Selective_Laser_Sintering", "Laminated_Object_Manufacturing",
]


class MockVLM(BaseVLM):
    """``image`` is the hidden ground-truth feature dict for the component."""

    def __init__(self, noise: Optional[NoiseConfig] = None, seed: int = 0):
        self.noise = noise or NoiseConfig()
        self.seed = seed
        self._idx: Dict[str, int] = defaultdict(int)  # channel -> next index

    def _next_index(self, channel: str) -> int:
        i = self._idx[channel]
        self._idx[channel] = i + 1
        return i

    def ask(self, image, question, feature_name, feature_type) -> VLMResponse:
        truth = image.get(feature_name) if image else None
        if truth is None:
            return VLMResponse("unknown", None, feature_name)

        idx = self._next_index(feature_name)
        r = rng(self.seed, feature_name, idx)

        if feature_type == "numeric":
            sigma = abs(truth) * self.noise.numeric_rel_noise + 1e-9
            val = round(max(0.0, float(truth) + r.gauss(0.0, sigma)), 3)
            return VLMResponse(f"approximately {val}", val, feature_name)

        if feature_type == "boolean":
            flip = r.random() < self.noise.boolean_flip_prob
            val = (not bool(truth)) if flip else bool(truth)
            return VLMResponse("yes" if val else "no", val, feature_name)

        if feature_type == "integer":
            span = self.noise.integer_swap_range
            val = max(0, int(truth) + r.randint(-span, span))
            return VLMResponse(str(val), val, feature_name)

        return VLMResponse(str(truth), truth, feature_name)

    def ask_process(self, image) -> str:
        idx = self._next_index("__process__")
        r = rng(self.seed, "__process__", idx)
        if r.random() < self.noise.inj_molding_prior:
            return "Injection_Molding"
        return r.choice(PROCESS_CHOICES)

    def ask_process_topk(self, image, k: int) -> list:
        """Biased top-k ranking: the (biased) single guess first, then the
        remaining processes with Injection_Molding promoted to the front --
        the ranked analogue of the corpus-frequency prior -- followed by a
        seeded shuffle of the rest. Deterministic per (seed, call index)."""
        idx = self._next_index("__process_topk__")
        r = rng(self.seed, "__process_topk__", idx)
        first = ("Injection_Molding"
                 if r.random() < self.noise.inj_molding_prior
                 else r.choice(PROCESS_CHOICES))
        rest = [p for p in PROCESS_CHOICES if p != first]
        r.shuffle(rest)
        if first != "Injection_Molding" and "Injection_Molding" in rest:
            rest.remove("Injection_Molding")
            rest.insert(0, "Injection_Molding")
        return ([first] + rest)[:max(1, k)]
