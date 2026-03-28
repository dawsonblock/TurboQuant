"""
Tests for maybe_turboquant_k_cache() and generate_step turboquant kwargs.

Coverage (from test-plan doc):
  1. noop_when_threshold_is_none        – no upgrade when turboquant_k_start=None
  2. noop_before_threshold              – no upgrade while offset < threshold
  3. upgrades_at_threshold              – KVCache → TurboQuantKCache once offset == threshold
  4. preserves_offset_and_state_shape   – offset and key shape survive upgrade
  5. idempotent                         – calling hook twice does not double-wrap
  6. upgraded_cache_accepts_more_tokens – update_and_fetch still works after upgrade
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import mlx.core as mx
from mlx_lm.models.cache import KVCache, TurboQuantKCache
from mlx_lm.generate import maybe_turboquant_k_cache


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _make_dense_prompt_cache(offset: int, heads: int = 2, head_dim: int = 8):
    """Build a list[KVCache] whose .keys/.values carry `offset` filled tokens."""
    cache = KVCache()  # KVCache takes no init args; shape inferred on first use
    if offset > 0:
        dummy_k = mx.zeros((1, heads, offset, head_dim))
        dummy_v = mx.zeros((1, heads, offset, head_dim))
        cache.update_and_fetch(dummy_k, dummy_v)
    return [cache]


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_noop_when_threshold_is_none():
    """maybe_turboquant_k_cache is a no-op when turboquant_k_start=None."""
    prompt_cache = _make_dense_prompt_cache(offset=20)
    original = prompt_cache[0]

    maybe_turboquant_k_cache(
        prompt_cache,
        turboquant_k_start=None,
        turboquant_main_bits=3,
        turboquant_group_size=64,
        turboquant_rotation="identity",
        turboquant_return_mode="dequant",
        turboquant_resid_scale_bits=8,
        turboquant_v_bits=4,
        turboquant_v_group_size=64,
        turboquant_v_enabled=True,
    )

    assert prompt_cache[0] is original, "Cache should not change when threshold is None"
    assert isinstance(prompt_cache[0], KVCache)


def test_noop_before_threshold():
    """No upgrade while current offset < turboquant_k_start."""
    prompt_cache = _make_dense_prompt_cache(offset=10)

    maybe_turboquant_k_cache(
        prompt_cache,
        turboquant_k_start=20,
        turboquant_main_bits=3,
        turboquant_group_size=64,
        turboquant_rotation="identity",
        turboquant_return_mode="dequant",
        turboquant_resid_scale_bits=8,
        turboquant_v_bits=4,
        turboquant_v_group_size=64,
        turboquant_v_enabled=True,
    )

    assert isinstance(prompt_cache[0], KVCache), (
        "Cache should remain KVCache when below threshold"
    )


def test_upgrades_at_threshold():
    """KVCache is replaced with TurboQuantKCache once offset >= turboquant_k_start."""
    threshold = 16
    prompt_cache = _make_dense_prompt_cache(offset=threshold)

    maybe_turboquant_k_cache(
        prompt_cache,
        turboquant_k_start=threshold,
        turboquant_main_bits=3,
        turboquant_group_size=8,   # small so head_dim=8 is divisible
        turboquant_rotation="identity",
        turboquant_return_mode="dequant",
        turboquant_resid_scale_bits=8,
        turboquant_v_bits=4,
        turboquant_v_group_size=8,
        turboquant_v_enabled=True,
    )

    assert isinstance(prompt_cache[0], TurboQuantKCache), (
        "Cache should be upgraded to TurboQuantKCache at threshold"
    )


def test_preserves_offset_and_state_shape():
    """After upgrade the TurboQuantKCache offset matches original and keys exist."""
    threshold = 16
    prompt_cache = _make_dense_prompt_cache(offset=threshold)
    original_offset = prompt_cache[0].offset  # == threshold

    maybe_turboquant_k_cache(
        prompt_cache,
        turboquant_k_start=threshold,
        turboquant_main_bits=3,
        turboquant_group_size=8,
        turboquant_rotation="identity",
        turboquant_return_mode="dequant",
        turboquant_resid_scale_bits=8,
        turboquant_v_bits=4,
        turboquant_v_group_size=8,
        turboquant_v_enabled=True,
    )

    tq = prompt_cache[0]
    assert isinstance(tq, TurboQuantKCache)
    assert tq.offset == original_offset, (
        f"Offset should be preserved: got {tq.offset}, expected {original_offset}"
    )
    # keys storage should be populated
    assert tq.k_codes is not None, "Quantized key codes should exist after migration"


def test_idempotent():
    """Calling maybe_turboquant_k_cache twice does not double-wrap the cache."""
    threshold = 16
    prompt_cache = _make_dense_prompt_cache(offset=threshold)

    kwargs = dict(
        turboquant_k_start=threshold,
        turboquant_main_bits=3,
        turboquant_group_size=8,
        turboquant_rotation="identity",
        turboquant_return_mode="dequant",
        turboquant_resid_scale_bits=8,
        turboquant_v_bits=4,
        turboquant_v_group_size=8,
        turboquant_v_enabled=True,
    )

    maybe_turboquant_k_cache(prompt_cache, **kwargs)
    first = prompt_cache[0]
    assert isinstance(first, TurboQuantKCache)

    maybe_turboquant_k_cache(prompt_cache, **kwargs)
    assert prompt_cache[0] is first, "Second call should return the same TurboQuantKCache"
    assert isinstance(prompt_cache[0], TurboQuantKCache)


def test_upgraded_cache_accepts_more_tokens():
    """After upgrade, update_and_fetch still works for new decode tokens."""
    threshold = 16
    heads, head_dim = 2, 8
    prompt_cache = _make_dense_prompt_cache(
        offset=threshold, heads=heads, head_dim=head_dim
    )

    maybe_turboquant_k_cache(
        prompt_cache,
        turboquant_k_start=threshold,
        turboquant_main_bits=3,
        turboquant_group_size=8,
        turboquant_rotation="identity",
        turboquant_return_mode="dequant",
        turboquant_resid_scale_bits=8,
        turboquant_v_bits=4,
        turboquant_v_group_size=8,
        turboquant_v_enabled=True,
    )

    tq = prompt_cache[0]
    assert isinstance(tq, TurboQuantKCache)

    # decode one more token
    new_k = mx.zeros((1, heads, 1, head_dim))
    new_v = mx.zeros((1, heads, 1, head_dim))
    result_k, result_v = tq.update_and_fetch(new_k, new_v)

    assert tq.offset == threshold + 1, (
        f"Offset should be {threshold + 1} after one decode step, got {tq.offset}"
    )
    # result_k carries at least the new token
    assert result_k is not None
    assert result_v is not None
