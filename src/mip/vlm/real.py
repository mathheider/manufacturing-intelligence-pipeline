"""Real VLM backends (Gemini, Claude). Imported lazily by ``get_backend`` so the
offline benchmark never needs their dependencies. Each receives an image path
(or PIL image) and returns the same VLMResponse the MockVLM does, so the agent
is entirely unaware which backend it is driving.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional

from .base import BaseVLM, VLMResponse
from .mock import PROCESS_CHOICES

_SUFFIX = {
    "numeric": " Reply with only a single number (no units, no words). If unknown, reply 'unknown'.",
    "boolean": " Reply with exactly one word: yes, no, or unknown.",
    "integer": " Reply with a single non-negative integer. If unknown, reply 'unknown'.",
}
_PROCESS_PROMPT = (
    "You are a polymer-composite manufacturing expert. Which single process is "
    "most likely used to manufacture the part in this image? Choose exactly one "
    "of: {choices}. Reply with only the process name, nothing else."
)
_PROCESS_TOPK_PROMPT = (
    "You are a polymer-composite manufacturing expert. Rank the {k} most likely "
    "manufacturing processes for the part in this image, most likely first. "
    "Choose only from: {choices}. Reply with exactly {k} process names, one per "
    "line, nothing else."
)


def parse(raw: str, feature_type: str) -> Any:
    if not raw or raw.lower().startswith("unknown"):
        return None
    if feature_type == "numeric":
        m = re.search(r"-?\d+(?:\.\d+)?", raw)
        return float(m.group()) if m else None
    if feature_type == "integer":
        m = re.search(r"-?\d+", raw)
        return int(m.group()) if m else None
    if feature_type == "boolean":
        low = raw.lower()
        if re.search(r"\b(yes|true|present)\b", low):
            return True
        if re.search(r"\b(no|false|absent|not)\b", low):
            return False
        return None
    return raw


def match_process(raw: str) -> str:
    norm = raw.replace(" ", "_").replace("-", "_").lower()
    for p in PROCESS_CHOICES:
        if p.lower() in norm:
            return p
    return "Injection_Molding"       # conservative bias-consistent fallback


def parse_topk(raw: str, k: int) -> list:
    """Parse a ranked list reply (one per line / comma separated) into k
    distinct process names; unparseable or missing ranks are filled from
    PROCESS_CHOICES order so the set size is always exactly k."""
    ranked: list = []
    for token in re.split(r"[\n,;]+", raw or ""):
        token = token.strip(" .:-0123456789)")
        if not token:
            continue
        p = match_process(token)
        if p not in ranked:
            ranked.append(p)
        if len(ranked) >= k:
            break
    for p in PROCESS_CHOICES:
        if len(ranked) >= k:
            break
        if p not in ranked:
            ranked.append(p)
    return ranked[:k]


class _RealVLM(BaseVLM):
    """Shared ask/ask_process; subclasses implement ``_call(image, prompt)``."""

    def _call(self, image, prompt: str) -> str:
        raise NotImplementedError

    def ask(self, image, question, feature_name, feature_type) -> VLMResponse:
        raw = self._call(image, f"{question}{_SUFFIX.get(feature_type, '')}")
        return VLMResponse(raw, parse(raw, feature_type), feature_name)

    def ask_process(self, image) -> str:
        raw = self._call(image, _PROCESS_PROMPT.format(choices=", ".join(PROCESS_CHOICES)))
        return match_process(raw)

    def ask_process_topk(self, image, k: int) -> list:
        raw = self._call(image, _PROCESS_TOPK_PROMPT.format(
            k=k, choices=", ".join(PROCESS_CHOICES)))
        return parse_topk(raw, k)


class GeminiVLM(_RealVLM):
    def __init__(self, model: Optional[str] = None, temperature: float = 0.7,
                 api_key: Optional[str] = None):
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise ImportError("GeminiVLM needs: pip install 'mip[gemini]'") from e
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("set GEMINI_API_KEY")
        genai.configure(api_key=key)
        self._genai = genai
        self._model = genai.GenerativeModel(model or os.environ.get("MIP_GEMINI_MODEL", "gemini-2.5-flash"))
        self._temperature = float(temperature)
        self._cache: dict = {}

    def _load(self, image):
        # 2026-08-05 infra fix: no memoization. The unbounded decoded-image
        # cache grew with every distinct photo across a 635-photo run and is
        # the prime suspect in the silent kill of the first container window.
        # The bytes sent per call are identical; decode cost is negligible
        # against network latency. No prompt/model/sampling change.
        if hasattr(image, "mode"):
            return image
        from PIL import Image
        img = Image.open(str(image)); img.load()
        return img

    def _call(self, image, prompt: str) -> str:
        import time
        cfg = self._genai.types.GenerationConfig(temperature=self._temperature, max_output_tokens=256)
        last = None
        for attempt in range(5):        # 429/5xx backoff; raise (never cache) on persistent failure
            try:
                resp = self._model.generate_content([prompt, self._load(image)],
                                                    generation_config=cfg)
                try:
                    return (resp.text or "").strip()
                except ValueError:
                    return "unknown"
            except Exception as e:      # noqa: BLE001 - SDK raises varied transient errors
                last = e
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"Gemini call failed after retries: {last}") from last


class ClaudeVLM(_RealVLM):
    DEFAULT_MODEL = "claude-sonnet-5"

    def __init__(self, model: Optional[str] = None, temperature: float = 0.7,
                 api_key: Optional[str] = None):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError("ClaudeVLM needs: pip install 'mip[claude]'") from e
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("set ANTHROPIC_API_KEY")
        self._client = anthropic.Anthropic(api_key=key)
        self._model = model or os.environ.get("MIP_CLAUDE_MODEL", self.DEFAULT_MODEL)
        self._temperature = float(temperature)
        self._cache: dict = {}

    def _b64(self, image):
        import base64
        if str(image) in self._cache:
            return self._cache[str(image)]
        path = Path(str(image))
        media = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                 "webp": "image/webp", "gif": "image/gif"}.get(path.suffix.lower().lstrip("."), "image/png")
        data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
        self._cache[str(image)] = (data, media)
        return data, media

    def _call(self, image, prompt: str) -> str:
        data, media = self._b64(image)
        msg = self._client.messages.create(
            model=self._model, max_tokens=256, temperature=self._temperature,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}},
                {"type": "text", "text": prompt}]}])
        return (msg.content[0].text or "").strip()
