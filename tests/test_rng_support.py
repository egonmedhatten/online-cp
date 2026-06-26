import numpy as np
import pytest

from online_cp import ConformalNearestNeighboursClassifier, ConformalRidgeRegressor
from online_cp.betting import ParticleFilterStrategy


def test_ridge_rng_int():
    # Ensure int seed still works
    cp1 = ConformalRidgeRegressor(rnd_state=42)
    cp2 = ConformalRidgeRegressor(rnd_state=42)
    # These should have the same RNG state initially
    assert np.all(cp1.rnd_gen.random(5) == cp2.rnd_gen.random(5))


def test_ridge_rng_generator():
    # Ensure Generator object works
    rng = np.random.default_rng(42)
    cp = ConformalRidgeRegressor(rnd_state=rng)
    assert cp.rnd_gen is rng
    # Consuming from cp.rnd_gen should advance the original rng
    val1 = rng.random()
    val2 = cp.rnd_gen.random()
    assert val1 != val2


def test_shared_rng_sequential():
    # Test that sharing a generator leads to sequential consumption, not cloning
    rng = np.random.default_rng(42)

    # CP1 and CP2 share the SAME generator object
    cp1 = ConformalRidgeRegressor(rnd_state=rng)
    cp2 = ConformalRidgeRegressor(rnd_state=rng)

    # First call to cp1
    res1 = cp1.rnd_gen.random()
    # Second call to cp2 should NOT be the same as res1
    res2 = cp2.rnd_gen.random()

    assert res1 != res2, "Shared RNG should be consumed sequentially, not cloned"


def test_particle_filter_rng():
    # Test the betting strategy
    rng = np.random.default_rng(123)
    pf = ParticleFilterStrategy(seed=rng)
    assert pf.rng is rng

    # Verify it actually uses the generator for particles
    # (This is just a smoke test to ensure the init doesn't crash)
    pf.update(0.1)


def test_classifier_rng():
    rng = np.random.default_rng(456)
    clf = ConformalNearestNeighboursClassifier(rnd_state=rng)
    assert clf.rnd_gen is rng
