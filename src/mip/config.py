"""One dataclass that holds every hyperparameter and seed. It is echoed into
every benchmark output so a result is never separated from the config that
produced it -- the first requirement of a reproducible experiment.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict


@dataclass
class NoiseConfig:
    """MockVLM perception-noise model."""

    numeric_rel_noise: float = 0.15     # relative gaussian noise on numerics
    boolean_flip_prob: float = 0.20     # P(flip) per boolean read
    integer_swap_range: int = 1         # +/- jitter on integer reads
    inj_molding_prior: float = 0.90     # P(direct-classify answers Inj. Molding)


@dataclass
class Config:
    seed: int = 20260630                # master seed; per-instance seeds derived
    n_samples: int = 5                  # consistency samples per feature probe
    tol_rel: float = 0.10               # numeric stability: max relative stdev
    min_agree: float = 0.60             # categorical stability: min majority frac
    max_steps: int = 20                 # agent probe budget per component
    max_reprobe: int = 1                # retries for an unstable feature, then abstain
    seeds: int = 40                     # MockVLM trials averaged per component
    noise: NoiseConfig = field(default_factory=NoiseConfig)

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, path: Path | str | None) -> "Config":
        if not path:
            return cls()
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        noise = NoiseConfig(**raw.pop("noise", {}))
        return cls(noise=noise, **raw)

    def to_dict(self) -> Dict:
        return asdict(self)

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
