"""Disk cache wrapper for real VLM backends (CLAUDE_CODE_GOALS.md goal 8).

Every raw VLM response is archived on disk, keyed by
``(image sha256, pipeline-prefixed channel, sample index, prompt version)``,
so a real-image benchmark run is:

* **resumable** -- re-running after an interruption replays cached responses
  byte-identically and only pays for calls it never made;
* **auditable** -- the raw text of every perception call survives in
  ``outputs/vlm_cache/bench/`` next to the question that produced it;
* **cheap to repeat** -- a re-run with the same pool and prompt version is
  free.

Replay determinism: within one benchmark run the pipelines execute in a fixed
order and each carries its own channel prefix (``direct:``, ``greedy:``,
``hybrid:``), so the i-th sample of a feature for a given image always maps to
the same cache file. Cached replies reproduce the same agent trajectory, which
reproduces the same cache lookups.

Bump PROMPT_VERSION whenever any perception prompt text changes -- old cache
entries then stop matching and the benchmark re-perceives.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .base import BaseVLM, VLMResponse

# bench-v1: templated boolean questions (archived 150-photo flash-lite run).
# bench-v2: explicit closed-hollow/undercut/cross-section/axisymmetric
#           questions after the provenance tally showed the templated
#           'hollow' phrasing caused 58% of false eliminations.
PROMPT_VERSION = "bench-v2"


class CachedVLM(BaseVLM):
    def __init__(self, inner: BaseVLM, cache_dir: Path | str,
                 prompt_version: str = PROMPT_VERSION):
        self.inner = inner
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.prompt_version = prompt_version
        self.prefix = ""                       # set per pipeline by the runner
        self.hits = 0
        self.misses = 0
        self._sha: dict = {}
        self._idx = defaultdict(int)           # (sha, channel) -> next index

    # -- helpers -------------------------------------------------------------

    def _image_sha(self, image) -> str:
        key = str(image)
        if key not in self._sha:
            self._sha[key] = hashlib.sha256(
                Path(key).read_bytes()).hexdigest()[:24]
        return self._sha[key]

    def _path(self, sha: str, channel: str, idx: int) -> Path:
        safe = channel.replace(":", "-").replace("/", "-")
        return self.dir / f"{sha}_{safe}_{idx:02d}_{self.prompt_version}.json"

    def _lookup(self, image, channel: str) -> tuple:
        sha = self._image_sha(image)
        idx = self._idx[(sha, channel)]
        self._idx[(sha, channel)] += 1
        return self._path(sha, channel, idx), sha

    def _store(self, path: Path, record: dict) -> None:
        record["model"] = str(getattr(self.inner, "_model", type(self.inner).__name__))
        path.write_text(json.dumps(record), encoding="utf-8")

    # -- BaseVLM interface ----------------------------------------------------

    def ask(self, image, question, feature_name, feature_type) -> VLMResponse:
        from .real import parse
        path, _ = self._lookup(image, f"{self.prefix}{feature_name}")
        if path.exists():
            self.hits += 1
            raw = json.loads(path.read_text(encoding="utf-8"))["raw"]
        else:
            self.misses += 1
            raw = self.inner.ask(image, question, feature_name, feature_type).raw
            self._store(path, {"raw": raw, "question": question,
                               "feature": feature_name, "type": feature_type})
        return VLMResponse(raw, parse(raw, feature_type), feature_name)

    def ask_process(self, image) -> str:
        from .real import match_process
        path, _ = self._lookup(image, f"{self.prefix}__process__")
        if path.exists():
            self.hits += 1
            raw = json.loads(path.read_text(encoding="utf-8"))["raw"]
        else:
            self.misses += 1
            raw_resp = self.inner.ask_process(image)
            # inner returns the matched name already; archive it as raw
            raw = raw_resp
            self._store(path, {"raw": raw, "question": "__process__"})
        return match_process(raw)

    def ask_process_topk(self, image, k: int) -> list:
        from .real import parse_topk
        path, _ = self._lookup(image, f"{self.prefix}__process_top{k}__")
        if path.exists():
            self.hits += 1
            raw = json.loads(path.read_text(encoding="utf-8"))["raw"]
            return parse_topk(raw, k)
        self.misses += 1
        ranked = self.inner.ask_process_topk(image, k)
        self._store(path, {"raw": "\n".join(ranked), "question": f"__process_top{k}__"})
        return ranked
