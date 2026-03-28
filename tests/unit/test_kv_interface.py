"""
Tests for turboquant.runtime.kv_interface — KVCompressor end-to-end.

Invariants verified
-------------------
* update_and_fetch returns (TurboQuantKeysView, values).
* iter_rotated_kv_blocks covers exactly the stored token range.
* K reconstruction error < 5 % relative (rotated space).
* V reconstruction error < 5 % relative.
* Multi-step accumulation works (offset advances correctly).
* state() / from_state() round-trip restores identical tensors.
* ensure_layout rejects non-4D inputs.
"""
import mlx.core as mx
import numpy as np
import pytest

from turboquant.config import TurboQuantConfig
from turboquant.runtime.kv_interface import KVCompressor, TurboQuantKeysView
from turboquant.runtime.layout import ensure_layout


def _rand(shape, seed=0):
    np.random.seed(seed)
    return mx.array(np.random.randn(*shape).astype(np.float32))


def _default_cfg(**kw):
    defaults = dict(
        k_bits=4, k_group_size=16,
        v_bits=4, v_group_size=16,
        rotation="hadamard",
        residual_topk=2,
        block_tokens=4,
        allocation_step=16,
    )
    defaults.update(kw)
    return TurboQuantConfig(**defaults)


# ── update_and_fetch ──────────────────────────────────────────────────────────

def test_update_returns_view_and_values():
    cfg = _default_cfg()
    cache = KVCompressor(cfg)
    k = _rand((1, 2, 4, 32))
    v = _rand((1, 2, 4, 32))
    view, vals = cache.update_and_fetch(k, v)
    assert isinstance(view, TurboQuantKeysView)
    assert vals.shape == v.shape
    assert cache.offset == 4


def test_multi_step_offset():
    cfg = _default_cfg()
    cache = KVCompressor(cfg)
    for t in [2, 3, 1]:
        k = _rand((1, 2, t, 32))
        v = _rand((1, 2, t, 32))
        cache.update_and_fetch(k, v)
    assert cache.offset == 6


# ── iter_rotated_kv_blocks ────────────────────────────────────────────────────

def test_block_iter_covers_all_tokens():
    cfg = _default_cfg(block_tokens=4)
    cache = KVCompressor(cfg)
    k = _rand((1, 2, 10, 32))
    v = _rand((1, 2, 10, 32))
    view, _ = cache.update_and_fetch(k, v)

    covered = set()
    for s, e, k_blk, v_blk in cache.iter_rotated_kv_blocks(view):
        covered.update(range(s, e))

    assert covered == set(range(10)), f"Not all tokens covered: {covered}"


def test_block_shapes():
    cfg = _default_cfg(block_tokens=4)
    cache = KVCompressor(cfg)
    k = _rand((1, 2, 8, 32))
    v = _rand((1, 2, 8, 32))
    view, _ = cache.update_and_fetch(k, v)

    for s, e, k_blk, v_blk in cache.iter_rotated_kv_blocks(view):
        T = e - s
        assert k_blk.shape == (1, 2, T, 32)
        assert v_blk.shape == (1, 2, T, 32)


# ── Reconstruction quality ────────────────────────────────────────────────────

def test_k_reconstruction_quality():
    cfg = _default_cfg(residual_topk=4)
    cache = KVCompressor(cfg)
    k = _rand((1, 2, 8, 32))
    v = _rand((1, 2, 8, 32))
    view, _ = cache.update_and_fetch(k, v)

    # Collect all decoded K blocks
    parts = []
    for _, _, k_blk, _ in cache.iter_rotated_kv_blocks(view):
        parts.append(k_blk)

    k_hat = mx.concatenate(parts, axis=2)          # [1,2,8,32] rotated
    k_rot = cache.pipeline.rotate_queries(k)        # expected rotated
    mx.eval(k_hat, k_rot)

    sig = float(mx.max(k_rot).item() - mx.min(k_rot).item()) + 1e-8
    err = float(mx.mean(mx.abs(k_rot - k_hat)).item()) / sig
    assert err < 0.05, f"K reconstruction error = {err:.4f} > 5 %"


def test_v_reconstruction_quality():
    cfg = _default_cfg(residual_topk=0)
    cache = KVCompressor(cfg)
    k = _rand((1, 2, 8, 32))
    v = _rand((1, 2, 8, 32))
    view, _ = cache.update_and_fetch(k, v)

    parts = []
    for _, _, _, v_blk in cache.iter_rotated_kv_blocks(view):
        parts.append(v_blk)

    v_hat = mx.concatenate(parts, axis=2)
    mx.eval(v, v_hat)

    sig = float(mx.max(v).item() - mx.min(v).item()) + 1e-8
    err = float(mx.mean(mx.abs(v - v_hat)).item()) / sig
    assert err < 0.05, f"V reconstruction error = {err:.4f} > 5 %"


# ── State serialisation ───────────────────────────────────────────────────────

def test_state_restore():
    cfg = _default_cfg()
    cache = KVCompressor(cfg)
    k = _rand((1, 2, 6, 32))
    v = _rand((1, 2, 6, 32))
    cache.update_and_fetch(k, v)

    st = cache.state()
    cache2 = KVCompressor.from_state(st, cfg)
    assert cache2.offset == cache.offset

    # Both should produce the same decoded K
    view1 = cache._make_view()
    view2 = cache2._make_view()

    blocks1 = list(cache.iter_rotated_kv_blocks(view1))
    blocks2 = list(cache2.iter_rotated_kv_blocks(view2))

    for (s1, e1, k1, v1), (s2, e2, k2, v2) in zip(blocks1, blocks2):
        mx.eval(k1, k2)
        diff = mx.max(mx.abs(k1 - k2)).item()
        assert diff < 1e-5, f"State restore mismatch at [{s1}:{e1}]: {diff}"


# ── Layout enforcement ────────────────────────────────────────────────────────

def test_ensure_layout_rejects_3d():
    x = mx.zeros((2, 4, 8))
    with pytest.raises(ValueError, match="4-D"):
        ensure_layout(x)


def test_ensure_layout_accepts_4d():
    x = mx.zeros((1, 2, 4, 16))
    assert ensure_layout(x) is x


def test_update_rejects_3d_keys():
    cfg = _default_cfg()
    cache = KVCompressor(cfg)
    with pytest.raises(ValueError):
        cache.update_and_fetch(mx.zeros((2, 4, 8)), mx.zeros((1, 2, 4, 16)))
