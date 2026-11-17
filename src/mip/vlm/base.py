"""VLM abstraction. Every backend (mock, Gemini, Claude, ...) implements the
same two-method interface, so the agent / planner / sampler are entirely
backbone-agnostic. Perception is the *only* thing a backend does; it never sees
the rules or the candidate set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class VLMResponse:
    raw: str
    parsed: Any                 # typed value, or None on parse failure
    feature_name: str


class BaseVLM:
    def ask(self, image, question: str, feature_name: str,
            feature_type: str) -> VLMResponse:
        """Perceive one feature. ``parsed`` is a float/int/bool or None."""
        raise NotImplementedError

    def ask_process(self, image) -> str:
        """Single-pass direct process classification (the RAG+LLM baseline)."""
        raise NotImplementedError

    def ask_process_topk(self, image, k: int) -> list:
        """Direct classification returning the model's top-k ranked guesses.

        Used by the set-size-matched Direct@k ablation mode (goal 3), so the
        Direct baseline can emit a set comparable to Hybrid's survivor set
        instead of a singleton. Backends override; the default degrades to the
        single guess.
        """
        return [self.ask_process(image)][:k]
