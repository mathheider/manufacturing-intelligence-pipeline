"""The dataset's core invariant: every expert-feasible process must survive the
oracle rule engine under perfect perception. If it did not, recall would be
capped by mislabeled data rather than by perception -- and the benchmark would
be measuring the wrong thing.
"""

from mip.dataset import generate, load_seed
from mip.engine import eliminate
from mip.rules import RuleSet


def test_seed_components_are_rule_feasible():
    rules = RuleSet.load()
    for inst in load_seed():
        survivors, _ = eliminate(rules, inst.features)
        missing = inst.expert_processes - set(survivors)
        assert not missing, (
            f"{inst.id}: expert processes eliminated under perfect "
            f"perception: {sorted(missing)}")


def test_synthetic_instances_preserve_feasibility():
    rules = RuleSet.load()
    for inst in generate(count=300, seed=1):
        survivors, _ = eliminate(rules, inst.features)
        assert inst.expert_processes.issubset(set(survivors)), (
            f"{inst.id}: synthetic perturbation broke feasibility")


def test_synthetic_generation_is_deterministic():
    a = generate(count=50, seed=7)
    b = generate(count=50, seed=7)
    assert [x.features for x in a] == [x.features for x in b]


def test_synthetic_preserves_class_boolean_identity():
    # class-defining booleans must be held fixed across the expansion
    for inst in generate(count=100, seed=3):
        arch = {i.id: i for i in load_seed()}[inst.archetype]
        for b in ("hollow", "axisymmetric", "constant_cross_section", "undercut_present"):
            assert inst.features[b] == arch.features[b]
