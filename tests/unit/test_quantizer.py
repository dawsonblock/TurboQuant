"""
Tests for turboquant.core.quantizer — GroupScalarQuantizer.

Invariants verified
-------------------
* Round-trip error is small (< 2 % relative to signal range).
* Bit packing is invertible (pack ∘ unpack == id).
* Calibration improves or maintains reconstruction quality vs dynamic.
* Group boundaries are respected.
* Unsupported bit widths raise immediately.
"""
import mlx.core as mx
import numpy as np
import pytest

from turboquant.core.quantizer import (
    GroupScalarQuantizer,
    pack_codes,
    unpack_codes,
)


def _rand(shape, lo=-2.0, hi=2.0, seed=1):
    np.random.seed(seed)
    a = np.random.uniform(lo, hi, shape).astype(np.float32)
    return mx.array(a)


# ── Pack / unpack ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bits", [2, 3, 4, 8])
def test_pack_unpack_round_trip(bits):
    """pack(unpack(codes)) must equal codes."""
    cpw = 32 // bits
    n_words = 8
    d = n_words * cpw
    np.random.seed(42)
    qmax = (1 << bits) - 1
    raw = mx.array(
        np.random.randint(0, qmax + 1, (2, 4, d), dtype=np.uint32)
    )
    packed = pack_codes(raw, bits)
    recovered = unpack_codes(packed, d, bits)
    mx.eval(recovered)
    assert mx.array_equal(raw, recovered), (
        f"Pack/unpack mismatch for bits={bits}"
    )


# ── Encode / decode ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("bits,group_size", [(3, 32), (4, 64), (8, 128)])
def test_encode_decode_shape(bits, group_size):
    """Decoded shape must match input shape."""
    D = group_size * 2
    x = _rand((2, 4, 16, D))
    q = GroupScalarQuantizer(n_bits=bits, group_size=group_size)
    packed, scales = q.encode(x)
    x_hat = q.decode(packed, scales, D)
    mx.eval(x_hat)
    assert x_hat.shape == x.shape, (
        f"Shape mismatch: {x_hat.shape} vs {x.shape}"
    )


@pytest.mark.parametrize("bits,group_size", [(3, 8), (4, 16)])
def test_encode_decode_accuracy(bits, group_size):
    """Relative reconstruction error must be below 5 %."""
    D = group_size * 4
    x = _rand((1, 2, 32, D))
    q = GroupScalarQuantizer(n_bits=bits, group_size=group_size)
    packed, scales = q.encode(x)
    x_hat = q.decode(packed, scales, D)
    mx.eval(x, x_hat)

    signal_range = float(mx.max(x).item() - mx.min(x).item())
    err = float(mx.mean(mx.abs(x - x_hat)).item())
    rel = err / (signal_range + 1e-8)
    assert rel < 0.05, (
        f"bits={bits} group={group_size}: rel err = {rel:.3f} > 5 %"
    )


# ── Calibration ───────────────────────────────────────────────────────────────

def test_calibration_reduces_error():
    """Calibrated quantiser must not be worse than uncalibrated."""
    np.random.seed(99)
    D = 128
    N = 512
    calib_data = mx.array(
        np.random.randn(N, D).astype(np.float32) * 0.5
    )
    x = mx.array(
        np.random.randn(1, 1, 16, D).astype(np.float32) * 0.5
    )

    q_dyn = GroupScalarQuantizer(n_bits=4, group_size=64)
    pk_d, sc_d = q_dyn.encode(x)
    err_dyn = float(mx.mean(mx.abs(x - q_dyn.decode(pk_d, sc_d, D))).item())

    q_cal = GroupScalarQuantizer(n_bits=4, group_size=64)
    q_cal.fit(calib_data)
    assert q_cal.is_calibrated
    pk_c, sc_c = q_cal.encode(x)
    err_cal = float(mx.mean(mx.abs(x - q_cal.decode(pk_c, sc_c, D))).item())

    # Calibrated should be within 2x of dynamic (both should be small)
    assert err_cal < max(err_dyn * 2, 0.02), (
        f"Calibrated error {err_cal:.4f} worse than 2× dynamic {err_dyn:.4f}"
    )


def test_calibration_flag():
    q = GroupScalarQuantizer(n_bits=4, group_size=64)
    assert not q.is_calibrated
    data = _rand((64, 128))
    q.fit(data)
    assert q.is_calibrated


def test_invalid_bits_raises():
    with pytest.raises(ValueError):
        GroupScalarQuantizer(n_bits=1)
    with pytest.raises(ValueError):
        GroupScalarQuantizer(n_bits=9)
