"""Stable, cross-run deterministic seed derivation.

Python's built-in ``hash()`` is salted per process (PYTHONHASHSEED), so it must
never be used to derive experiment seeds. We use SHA-256 over the string form of
the components instead, which is identical on every machine and every run.
"""

from __future__ import annotations

import hashlib
import random


def derive_seed(*parts) -> int:
    """A reproducible 63-bit integer seed from any set of parts."""
    key = "|".join(str(p) for p in parts)
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def rng(*parts) -> random.Random:
    """A ``random.Random`` seeded deterministically from ``parts``."""
    return random.Random(derive_seed(*parts))
