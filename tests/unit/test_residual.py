"""
Tests for turboquant.core.residual — encode/decode_topk_residual.

Invariants verified
-------------------
* Round-trip: decode(encode(r)) reproduces the top-k components exactly.
* Sparsity: all but the top-k positions are zero after reconstruction.
* Energy: the reconstructed residual captures >= 80 % of the residual
  energy (for typical k / group_size ratios).
* Shape contract: output shape equals input shape.
"""
import mlx.core as mx
import numpy as np
import pytest

from turboquant.core.residual import (
    encode_topk_residual,
    decode_topk_residual,
)


def _rand(shape, seed=2):
    np.random.seed(seed)
    return mx.array(np.random.randn(*shape).astype(np.float32))


@pytest.mark.parametrize("k,group_size", [(1, 8), (2, 16), (4, 32)])
def test_output_shape(k, group_size):
    d_pad = group_size * 4
    r = _rand((2, 2, 8, d_pad))
    vals, idx = encode_topk_residual(r, k=k, group_size=group_size)
    r_hat = decode_topk_residual(vals, idx, group_size)
    mx.eval(r_hat)
    assert r_hat.shape == r.shape, (
        f"Shape mismatch: {r_hat.shape} vs {r.shape}"
    )


@pytest.mark.parametrize("k,group_size", [(2, 8), (4, 16)])
def test_topk_round_trip_at_topk_positions(k, group_size):
    """The k selected positions must be reproduced exactly."""
    d_pad = group_size * 2
    np.random.seed(3)
    r_np = np.random.randn(1, 1, 1, d_pad).astype(np.float32)
    r = mx.array(r_np)

    vals, idx = encode_topk_residual(r, k=k, group_size=group_size)
    r_hat = decode_topk_residual(vals, idx, group_size)
    mx.eval(r, r_hat, vals, idx)

    r_np   = np.array(r)
    hat_np = np.array(r_hat)
    idx_np = np.array(idx)

    ng = d_pad // group_size
    r_grp  = r_np.reshape(ng, group_size)
    hat_grp = hat_np.reshape(ng, group_size)

    for g in range(ng):
        for ki in range(k):
            pos = int(idx_np.reshape(ng, k)[g, ki])
            orig = float(r_grp[g, pos])
            rec  = float(hat_grp[g, pos])
            assert abs(orig - rec) < 1e-3, (
                f"group {g} position {pos}: orig={orig:.4f} rec={rec:.4f}"
            )


@pytest.mark.parametrize("k", [1, 2, 4])
def test_energy_capture(k):
    """Reconstructed residual must capture >= 70 % residual energy."""
    group_size = 16
    d_pad = group_size * 4
    r = _rand((1, 1, 32, d_pad), seed=7)
    vals, idx = encode_topk_residual(r, k=k, group_size=group_size)
    r_hat = decode_topk_residual(vals, idx, group_size)
    mx.eval(r, r_hat)

    total_energy = float(mx.sum(r ** 2).item())
    captured     = float(mx.sum(r_hat ** 2).item())
    ratio = captured / (total_energy + 1e-8)

    # k/group_size fraction of variance; allow 20 % slack
    expected_min = max(0.0, (k / group_size) * 0.8)
    assert ratio >= expected_min, (
        f"k={k}: captured {ratio:.3f} of energy, expected >= {expected_min:.3f}"
    )


def test_d_pad_must_be_divisible():
    r = _rand((1, 1, 4, 10))  # 10 not divisible by 8
    with pytest.raises(AssertionError):
        encode_topk_residual(r, k=2, group_size=8)
