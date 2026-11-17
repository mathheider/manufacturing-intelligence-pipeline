"""VLM backends. MockVLM is always available (stdlib only). Real backends are
imported lazily so the offline benchmark never requires their dependencies.
"""

from __future__ import annotations

from .base import BaseVLM, VLMResponse
from .mock import MockVLM

__all__ = ["BaseVLM", "VLMResponse", "MockVLM", "get_backend"]


def get_backend(name: str, **kwargs) -> BaseVLM:
    name = (name or "mock").lower()
    if name == "mock":
        return MockVLM(**kwargs)
    if name == "gemini":
        from .real import GeminiVLM
        return GeminiVLM(**kwargs)
    if name == "claude":
        from .real import ClaudeVLM
        return ClaudeVLM(**kwargs)
    raise ValueError(f"unknown VLM backend {name!r}")
