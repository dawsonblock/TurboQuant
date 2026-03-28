"""
Tests for turboquant.core.rotation — FixedRotation.

Invariants verified
-------------------
* Determinism: two instances with the same seed produce identical outputs.
* Orthogonality: R^T R ≈ I  (within fp32 tolerance).
* Round-trip: inverse(forward(x)) ≈ x.
* Identity type: forward/inverse are no-ops.
* Save / load: persisted matrix reproduces identical results.
"""
import mlx.core as mx
import numpy as np
import pytest

from turboquant.core.rotation import FixedRotation


DIM = 64
ATOL = 1e-4


def _rand(shape, seed=0):
    np.random.seed(seed)
    return mx.array(np.random.randn(*shape).astype(np.float32))


@pytest.mark.parametrize("rtype", ["hadamard", "random_orthogonal"])
def test_rotation_determinism(rtype):
    """Two instances with the same seed must produce the same matrix."""
    r1 = FixedRotation(DIM, seed=42, rotation_type=rtype)
    r2 = FixedRotation(DIM, seed=42, rotation_type=rtype)
    mx.eval(r1._R, r2._R)
    diff = mx.max(mx.abs(r1._R - r2._R)).item()
    assert diff < 1e-7, f"Mismatch for {rtype}: max diff = {diff}"


def test_rotation_different_seeds_random_orthogonal():
    """random_orthogonal matrices must differ across seeds."""
    r1 = FixedRotation(DIM, seed=1, rotation_type="random_orthogonal")
    r2 = FixedRotation(DIM, seed=2, rotation_type="random_orthogonal")
    mx.eval(r1._R, r2._R)
    diff = mx.max(mx.abs(r1._R - r2._R)).item()
    assert diff > 1e-3, "Seeds 1/2 gave same matrix for random_orthogonal"


def test_hadamard_is_seed_independent():
    """Hadamard must be identical regardless of seed (it is fixed)."""
    r1 = FixedRotation(DIM, seed=1, rotation_type="hadamard")
    r2 = FixedRotation(DIM, seed=99, rotation_type="hadamard")
    mx.eval(r1._R, r2._R)
    diff = mx.max(mx.abs(r1._R - r2._R)).item()
    assert diff < 1e-7, "Hadamard matrices differ across seeds"


@pytest.mark.parametrize("rtype", ["hadamard", "random_orthogonal"])
def test_rotation_orthogonality(rtype):
    """R^T R should be close to identity."""
    rot = FixedRotation(DIM, seed=42, rotation_type=rtype)
    R = rot._R
    RtR = R.T @ R
    eye = mx.eye(DIM)
    mx.eval(RtR, eye)
    err = mx.max(mx.abs(RtR - eye)).item()
    assert err < ATOL, f"Orthogonality violation for {rtype}: {err:.2e}"


@pytest.mark.parametrize("rtype", ["identity", "hadamard", "random_orthogonal"])
def test_rotation_round_trip(rtype):
    """inverse(forward(x)) must recover x."""
    rot = FixedRotation(DIM, seed=42, rotation_type=rtype)
    x = _rand((2, 4, 16, DIM))
    y = rot.forward(x)
    x_rec = rot.inverse(y)
    mx.eval(x_rec)
    err = mx.max(mx.abs(x - x_rec)).item()
    assert err < ATOL, (
        f"Round-trip error for {rtype}: max abs err = {err:.2e}"
    )


def test_identity_is_noop():
    """Identity rotation must return the exact same array object."""
    rot = FixedRotation(DIM, rotation_type="identity")
    x = _rand((1, 2, 8, DIM))
    assert rot.forward(x) is x
    assert rot.inverse(x) is x


@pytest.mark.parametrize("rtype", ["hadamard", "random_orthogonal"])
def test_save_load_round_trip(tmp_path, rtype):
    """Saved matrix must reproduce identical forward pass."""
    path = str(tmp_path / f"rot_{rtype}.npy")
    rot1 = FixedRotation(DIM, seed=7, rotation_type=rtype)
    rot1.save(path)

    rot2 = FixedRotation.load(path)
    x = _rand((1, 1, 4, DIM))
    y1 = rot1.forward(x)
    y2 = rot2.forward(x)
    mx.eval(y1, y2)
    diff = mx.max(mx.abs(y1 - y2)).item()
    assert diff < 1e-7, f"Save/load mismatch for {rtype}: {diff:.2e}"


def test_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown rotation_type"):
        FixedRotation(DIM, rotation_type="fft")
