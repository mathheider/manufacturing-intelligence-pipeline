"""Consistency sampling and MockVLM determinism / fairness."""

from mip.config import NoiseConfig
from mip.sampling import sample_categorical, sample_numeric, run_consistency
from mip.vlm import MockVLM


def test_numeric_stability():
    assert sample_numeric([10.0, 10.1, 9.9], tol_rel=0.10).stable is True
    assert sample_numeric([10.0, 20.0, 5.0], tol_rel=0.10).stable is False
    assert sample_numeric([]).stable is False


def test_categorical_majority():
    assert sample_categorical([True, True, True, False], min_agree=0.6).stable is True
    assert sample_categorical([True, False, True, False], min_agree=0.6).stable is False


def test_mock_is_deterministic_across_instances():
    # Two fresh MockVLMs with the same seed give identical reads for the same
    # feature's same-index query, regardless of call interleaving.
    img = {"wall_thickness_mm": 3.0, "hollow": True}
    a = MockVLM(seed=42)
    b = MockVLM(seed=42)
    for _ in range(5):
        ra = a.ask(img, "q", "wall_thickness_mm", "numeric")
        rb = b.ask(img, "q", "wall_thickness_mm", "numeric")
        assert ra.parsed == rb.parsed


def test_mock_feature_independence():
    # Reading a different feature in between must not shift another feature's
    # per-index stream (the property that makes the ablation fair).
    img = {"wall_thickness_mm": 3.0, "hollow": True}
    a = MockVLM(seed=1)
    b = MockVLM(seed=1)
    a_first = a.ask(img, "q", "wall_thickness_mm", "numeric").parsed
    b.ask(img, "q", "hollow", "boolean")             # interleave a different read
    b_first = b.ask(img, "q", "wall_thickness_mm", "numeric").parsed
    assert a_first == b_first


def test_mock_process_bias():
    img = {"wall_thickness_mm": 3.0}
    counts = {"Injection_Molding": 0, "other": 0}
    for s in range(400):
        p = MockVLM(noise=NoiseConfig(inj_molding_prior=0.9), seed=s).ask_process(img)
        counts["Injection_Molding" if p == "Injection_Molding" else "other"] += 1
    # ~90% should be Injection_Molding
    assert counts["Injection_Molding"] / 400 > 0.8


def test_run_consistency_uses_valid_samples_only():
    img = {"wall_thickness_mm": 3.0}
    res = run_consistency(MockVLM(seed=0), img, "q", "wall_thickness_mm", "numeric", n=5)
    assert res.n_samples == 5
    assert res.value is not None
