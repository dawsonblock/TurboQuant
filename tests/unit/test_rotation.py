"""
Tests for turboquant.core.rotation — FixedRotation.

Invariants verified
-------------------
* Determinism: two instances with the same seed produce identical outputs.
* Orthogonality: R^T R ≈ I  (within fp32 tolerance).
* Round-trip: inverse(forward(x)) ≈ x.
* Identity type: forward/inverse are no-ops.
* Save / load: persisted matrix reproduces identical results.
* Hadamard path stays orthogonal for non-power-of-two dimensions.
"""
import mlx.core as mx
import numpy as np
import pytest

from turboquant.core.rotation import FixedRotation


ATOL = 1e-4


def _rand(shape, seed=0):
    np.random.seed(seed)
    return mx.array(np.random.randn(*shape).astype(np.float32))


@pytest.mark.parametrize("dim,rtype", [
    (64, "hadamard"),
    (64, "random_orthogonal"),
    (80, "hadamard"),
    (80, "random_orthogonal"),
])
def test_rotation_determinism(dim, rtype):
    """Two instances with the same seed must produce the same matrix."""
    r1 = FixedRotation(dim, seed=42, rotation_type=rtype)
    r2 = FixedRotation(dim, seed=42, rotation_type=rtype)
    mx.eval(r1._R, r2._R)
    diff = mx.max(mx.abs(r1._R - r2._R)).item()
    assert diff < 1e-7, f"Mismatch for {rtype} dim={dim}: max diff = {diff}"


def test_rotation_different_seeds_random_orthogonal():
    """random_orthogonal matrices must differ across seeds."""
    r1 = FixedRotation(64, seed=1, rotation_type="random_orthogonal")
    r2 = FixedRotation(64, seed=2, rotation_type="random_orthogonal")
    mx.eval(r1._R, r2._R)
    diff = mx.max(mx.abs(r1._R - r2._R)).item()
    assert diff > 1e-3, "Seeds 1/2 gave same matrix for random_orthogonal"


def test_hadamard_is_seed_independent():
    """Hadamard-derived rotations must be identical regardless of seed."""
    r1 = FixedRotation(80, seed=1, rotation_type="hadamard")
    r2 = FixedRotation(80, seed=99, rotation_type="hadamard")
    mx.eval(r1._R, r2._R)
    diff = mx.max(mx.abs(r1._R - r2._R)).item()
    assert diff < 1e-7, "Hadamard matrices differ across seeds"


@pytest.mark.parametrize("dim,rtype", [
    (64, "hadamard"),
    (64, "random_orthogonal"),
    (80, "hadamard"),
    (80, "random_orthogonal"),
])
def test_rotation_orthogonality(dim, rtype):
    """R^T R should be close to identity."""
    rot = FixedRotation(dim, seed=42, rotation_type=rtype)
    R = rot._R
    RtR = R.T @ R
    eye = mx.eye(dim)
    mx.eval(RtR, eye)
    err = mx.max(mx.abs(RtR - eye)).item()
    assert err < ATOL, f"Orthogonality violation for {rtype} dim={dim}: {err:.2e}"


@pytest.mark.parametrize("dim,rtype", [
    (64, "identity"),
    (64, "hadamard"),
    (64, "random_orthogonal"),
    (80, "hadamard"),
])
def test_rotation_round_trip(dim, rtype):
    """inverse(forward(x)) must recover x."""
    rot = FixedRotation(dim, seed=42, rotation_type=rtype)
    x = _rand((2, 4, 16, dim))
    y = rot.forward(x)
    x_rec = rot.inverse(y)
    mx.eval(x_rec)
    err = mx.max(mx.abs(x - x_rec)).item()
    assert err < ATOL, (
        f"Round-trip error for {rtype} dim={dim}: max abs err = {err:.2e}"
    )


def test_identity_is_noop():
    """Identity rotation must return the exact same array object."""
    rot = FixedRotation(64, rotation_type="identity")
    x = _rand((1, 2, 8, 64))
    assert rot.forward(x) is x
    assert rot.inverse(x) is x


@pytest.mark.parametrize("dim,rtype", [(64, "hadamard"), (80, "hadamard"), (64, "random_orthogonal")])
def test_save_load_round_trip(tmp_path, dim, rtype):
    """Saved matrix must reproduce identical forward pass."""
    path = str(tmp_path / f"rot_{rtype}_{dim}.npy")
    rot1 = FixedRotation(dim, seed=7, rotation_type=rtype)
    rot1.save(path)

    rot2 = FixedRotation.load(path)
    x = _rand((1, 1, 4, dim))
    y1 = rot1.forward(x)
    y2 = rot2.forward(x)
    mx.eval(y1, y2)
    diff = mx.max(mx.abs(y1 - y2)).item()
    assert diff < 1e-7, f"Save/load mismatch for {rtype} dim={dim}: {diff:.2e}"


def test_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown rotation_type"):
        FixedRotation(64, rotation_type="fft")
