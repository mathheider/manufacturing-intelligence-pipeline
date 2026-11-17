"""MIP -- Manufacturing Intelligence Pipeline (clean rebuild).

A VLM-plus-symbolic-rules system that selects polymer-composite manufacturing
processes from component features. Perception (a VLM) only reads the part; a
validated symbolic rule engine makes the categorical decision against cited
forbidden_if clauses; per-feature provenance binds the two.

Public surface:
    RuleSet            -- load + validate the rule table
    Config             -- all hyperparameters and the master seed
    eliminate          -- pure rule engine
    load_dataset       -- seed or synthetic instances
    run_direct / run_ruleset / run_greedy / run_hybrid  -- the four pipelines
    run_benchmark      -- the deterministic three/four-way ablation
"""

from __future__ import annotations

from .config import Config, NoiseConfig
from .dataset import Instance, load_dataset, load_seed, generate
from .engine import check, eliminate
from .metrics import Score, score
from .pipelines import (PipelineResult, run_direct, run_greedy, run_hybrid,
                        run_ruleset, ALL_METHODS)
from .rules import RuleSet
from .state import ConstraintGraph, Elimination, FeatureObservation

__version__ = "1.0.0"

__all__ = [
    "Config", "NoiseConfig", "RuleSet", "eliminate", "check",
    "Instance", "load_dataset", "load_seed", "generate",
    "Score", "score", "PipelineResult", "ALL_METHODS",
    "run_direct", "run_ruleset", "run_greedy", "run_hybrid",
    "ConstraintGraph", "Elimination", "FeatureObservation",
    "__version__",
]
