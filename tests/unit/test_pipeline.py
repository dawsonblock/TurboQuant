"""
Tests for turboquant.core.pipeline — TurboQuantPipeline.

Invariants verified
-------------------
* K encode/decode (rotated) has < 5 % relative error.
* V encode/decode has < 5 % relative error.
* With residual (topk=2) error is lower than without (topk=0).
* Calibration path runs without error.
* rotate_queries returns correct shape.
"""
import mlx.core as mx
import numpy as np
import pytest

from turboquant.config import TurboQuantConfig
from turboquant.core.pipeline import TurboQuantPipeline


def _rand(shape, seed=5):
    np.random.seed(seed)
    return mx.array(np.random.randn(*shape).astype(np.float32))


def _rel_err(x, x_hat):
    mx.eval(x, x_hat)
    sig = float(mx.max(x).item() - mx.min(x).item()) + 1e-8
    return float(mx.mean(mx.abs(x - x_hat)).item()) / sig


@pytest.fixture
def base_cfg():
    return TurboQuantConfig(
        k_bits=4,
        k_group_size=16,
        v_bits=4,
        v_group_size=16,
        rotation="hadamard",
        residual_topk=0,
    )


@pytest.fixture
def resid_cfg():
    return TurboQuantConfig(
        k_bits=4,
        k_group_size=16,
        v_bits=4,
        v_group_size=16,
        rotation="hadamard",
        residual_topk=2,
    )


def test_encode_k_shape(base_cfg):
    pipe = TurboQuantPipeline(base_cfg)
    x = _rand((1, 2, 8, 64))
    pk, ks, rv, ri = pipe.encode_k(x)
    mx.eval(pk, ks)
    assert pk.ndim == 4
    assert ks.shape[:-1] == pk.shape[:-1]
    assert rv is None and ri is None


def test_k_round_trip_no_residual(base_cfg):
    pipe = TurboQuantPipeline(base_cfg)
    x = _rand((1, 2, 8, 64))
    pk, ks, rv, ri = pipe.encode_k(x)
    # Rotate x to compare in rotated space
    x_rot = pipe.rotate_queries(x)
    x_hat = pipe.decode_k_rotated(pk, ks, rv, ri)
    assert _rel_err(x_rot, x_hat) < 0.05


def test_k_round_trip_with_residual(resid_cfg):
    pipe = TurboQuantPipeline(resid_cfg)
    x = _rand((1, 2, 8, 64))
    pk, ks, rv, ri = pipe.encode_k(x)
    x_rot = pipe.rotate_queries(x)
    x_hat = pipe.decode_k_rotated(pk, ks, rv, ri)
    assert _rel_err(x_rot, x_hat) < 0.04


def test_residual_improves_accuracy():
    """Topk residual must reduce reconstruction error."""
    x = _rand((1, 2, 16, 64), seed=9)

    cfg0 = TurboQuantConfig(k_bits=3, k_group_size=16, residual_topk=0)
    cfg4 = TurboQuantConfig(k_bits=3, k_group_size=16, residual_topk=4)

    p0 = TurboQuantPipeline(cfg0)
    p4 = TurboQuantPipeline(cfg4)

    pk0, ks0, rv0, ri0 = p0.encode_k(x)
    err0 = _rel_err(p0.rotate_queries(x), p0.decode_k_rotated(pk0, ks0, rv0, ri0))

    pk4, ks4, rv4, ri4 = p4.encode_k(x)
    err4 = _rel_err(p4.rotate_queries(x), p4.decode_k_rotated(pk4, ks4, rv4, ri4))

    assert err4 < err0, (
        f"Residual did not help: err0={err0:.4f} err4={err4:.4f}"
    )


def test_v_round_trip(base_cfg):
    pipe = TurboQuantPipeline(base_cfg)
    v = _rand((1, 2, 8, 64))
    pv, vs = pipe.encode_v(v)
    v_hat = pipe.decode_v(pv, vs)
    assert _rel_err(v, v_hat) < 0.05


def test_rotate_queries_shape(base_cfg):
    pipe = TurboQuantPipeline(base_cfg)
    _ = pipe.encode_k(_rand((1, 2, 4, 64)))  # init dims
    q = _rand((1, 8, 1, 64))
    q_rot = pipe.rotate_queries(q)
    mx.eval(q_rot)
    assert q_rot.shape == q.shape


def test_fit_k_runs(base_cfg):
    pipe = TurboQuantPipeline(base_cfg)
    data = _rand((32, 64))
    pipe.fit_k(data)
    assert pipe._k_quant is not None and pipe._k_quant.is_calibrated


def test_fit_v_runs(base_cfg):
    pipe = TurboQuantPipeline(base_cfg)
    data = _rand((32, 64))
    pipe.fit_v(data)
    assert pipe._v_quant is not None and pipe._v_quant.is_calibrated
