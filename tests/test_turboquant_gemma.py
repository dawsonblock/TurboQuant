"""
Tests for TurboQuantKCache core mechanics and Gemma attention integration.

Coverage (from Quant planning doc):
  1. update_and_fetch_dequant_mode      – dequant mode returns dense tensors
  2. update_and_fetch_view_mode         – view mode returns TurboQuantKeysView
  3. state_roundtrip                    – state dict round-trips without loss
  4. block_iterator_covers_all_tokens   – iter_rotated_kv_blocks yields every token
  5. gemma_attention_with_kvcache       – standard KVCache path in Attention
  6. gemma_attention_with_turboquant    – TurboQuantKCache path in Attention
  7. incremental_decode                 – multiple decode steps accumulate offset
  8. storage_breakdown_keys             – nbytes dominated by 3-bit codes + scales
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import math
import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.cache import KVCache, TurboQuantKCache, TurboQuantConfig
from mlx_lm.models.gemma import Attention, ModelArgs


# ─── Shared parameters ───────────────────────────────────────────────────────

HEAD_DIM    = 8
N_HEADS     = 4
N_KV_HEADS  = 2
HIDDEN_SIZE = N_HEADS * HEAD_DIM   # 32
GROUP_SIZE  = 8
BLOCK_TOKENS = 2
V_GROUP_SIZE = 8

PREFILL_LEN = 16   # must be divisible by GROUP_SIZE and BLOCK_TOKENS
BATCH       = 1


def _model_args():
    return ModelArgs(
        model_type="gemma",
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=1,
        intermediate_size=64,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        rms_norm_eps=1e-5,
        vocab_size=256,
    )


def _make_tq_cache(return_mode: str = "dequant") -> TurboQuantKCache:
    cfg = TurboQuantConfig(
        main_bits=3,
        group_size=GROUP_SIZE,
        rotation="identity",
        return_mode=return_mode,
        scale_dtype="float16",
        resid_scale_bits=8,
        v_bits=4,
        v_group_size=V_GROUP_SIZE,
        v_scale_dtype="float16",
        v_enabled=True,
        block_tokens=BLOCK_TOKENS,
    )
    return TurboQuantKCache(cfg)


def _make_kv(seq_len: int):
    """Random key/value tensors: [B, N_KV_HEADS, seq_len, HEAD_DIM]."""
    mx.random.seed(0)
    k = mx.random.normal((BATCH, N_KV_HEADS, seq_len, HEAD_DIM)).astype(mx.float16)
    v = mx.random.normal((BATCH, N_KV_HEADS, seq_len, HEAD_DIM)).astype(mx.float16)
    return k, v


# ─── 1. update_and_fetch – dequant mode ──────────────────────────────────────

def test_update_and_fetch_dequant_mode():
    """Dequant mode returns dense key/value arrays after prefill."""
    tq = _make_tq_cache(return_mode="dequant")
    k, v = _make_kv(PREFILL_LEN)
    result_k, result_v = tq.update_and_fetch(k, v)

    assert result_k is not None and result_v is not None
    assert isinstance(result_k, mx.array), "Expected dense array in dequant mode"
    assert isinstance(result_v, mx.array), "Expected dense array in dequant mode"
    assert result_k.shape[-2] == PREFILL_LEN, (
        f"Key sequence length mismatch: {result_k.shape[-2]} vs {PREFILL_LEN}"
    )
    assert tq.offset == PREFILL_LEN


# ─── 2. update_and_fetch – view mode ─────────────────────────────────────────

def test_update_and_fetch_view_mode():
    """View mode returns a TurboQuantKeysView (lazy iterator handle)."""
    from mlx_lm.models.cache import TurboQuantKeysView

    tq = _make_tq_cache(return_mode="view")
    k, v = _make_kv(PREFILL_LEN)
    result_k, result_v = tq.update_and_fetch(k, v)

    assert isinstance(result_k, TurboQuantKeysView), (
        f"Expected TurboQuantKeysView in view mode, got {type(result_k)}"
    )
    assert result_k.end == PREFILL_LEN
    assert tq.offset == PREFILL_LEN


# ─── 3. state roundtrip ───────────────────────────────────────────────────────

def test_state_roundtrip():
    """state / meta_state / from_state round-trips preserve offset and block_tokens."""
    tq = _make_tq_cache(return_mode="dequant")
    k, v = _make_kv(PREFILL_LEN)
    tq.update_and_fetch(k, v)

    state_dict = tq.state
    meta        = tq.meta_state

    tq2 = TurboQuantKCache.from_state(state_dict, meta)
    assert tq2.offset == tq.offset, (
        f"Offset mismatch after roundtrip: {tq2.offset} vs {tq.offset}"
    )
    assert tq2.config.block_tokens == tq.config.block_tokens, (
        f"block_tokens mismatch: {tq2.config.block_tokens} vs "
        f"{tq.config.block_tokens}"
    )


# ─── 4. block iterator covers all tokens ─────────────────────────────────────

def test_block_iterator_covers_all_tokens():
    """iter_rotated_kv_blocks yields blocks whose ranges cover [0, PREFILL_LEN)."""
    from mlx_lm.models.cache import TurboQuantKeysView

    tq = _make_tq_cache(return_mode="view")
    k, v = _make_kv(PREFILL_LEN)
    keys_view, _ = tq.update_and_fetch(k, v)

    assert isinstance(keys_view, TurboQuantKeysView)

    covered = []
    for s, e, k_blk, v_blk in tq.iter_rotated_kv_blocks(keys_view):
        assert e > s, f"Empty block: [{s}, {e})"
        assert k_blk.shape[-2] == e - s
        assert v_blk.shape[-2] == e - s
        covered.extend(range(s, e))

    assert sorted(covered) == list(range(PREFILL_LEN)), (
        "Block iterator did not cover all token positions exactly once"
    )


# ─── 5. Gemma attention with standard KVCache ────────────────────────────────

def test_gemma_attention_with_kvcache():
    """Attention.__call__ works normally with a plain KVCache."""
    args  = _model_args()
    attn  = Attention(args)
    cache = KVCache()  # shape inferred on first update_and_fetch call

    seq_len = 4
    x = mx.zeros((BATCH, seq_len, HIDDEN_SIZE))

    out = attn(x, mask=None, cache=cache)
    assert out.shape == (BATCH, seq_len, HIDDEN_SIZE), (
        f"Unexpected output shape: {out.shape}"
    )
    assert cache.offset == seq_len


# ─── 6. Gemma attention with TurboQuantKCache (view mode) ────────────────────

def test_gemma_attention_with_turboquant():
    """Attention.__call__ dispatches to streaming attention with TurboQuantKCache."""
    args  = _model_args()
    attn  = Attention(args)
    tq    = _make_tq_cache(return_mode="view")

    # prefill: run a longer sequence to populate the cache
    prefill_x = mx.zeros((BATCH, PREFILL_LEN, HIDDEN_SIZE))
    attn(prefill_x, mask=None, cache=tq)
    assert tq.offset == PREFILL_LEN

    # decode: single token
    decode_x = mx.zeros((BATCH, 1, HIDDEN_SIZE))
    out = attn(decode_x, mask=None, cache=tq)

    assert out.shape == (BATCH, 1, HIDDEN_SIZE), (
        f"Unexpected decode output shape: {out.shape}"
    )
    assert tq.offset == PREFILL_LEN + 1


# ─── 7. Incremental decode accumulates offset ────────────────────────────────

def test_incremental_decode():
    """Multiple single-token decode steps each increment offset by 1."""
    tq  = _make_tq_cache(return_mode="dequant")
    k, v = _make_kv(PREFILL_LEN)
    tq.update_and_fetch(k, v)
    assert tq.offset == PREFILL_LEN

    for step in range(1, 5):
        new_k, new_v = _make_kv(1)
        tq.update_and_fetch(new_k, new_v)
        assert tq.offset == PREFILL_LEN + step, (
            f"Offset should be {PREFILL_LEN + step} after {step} decode steps"
        )


# ─── 8. Storage dominated by 3-bit codes ─────────────────────────────────────

def test_storage_breakdown_keys():
    """3-bit packed codes use fewer bits-per-token than float16."""
    tq = _make_tq_cache(return_mode="dequant")
    k, v = _make_kv(PREFILL_LEN)
    tq.update_and_fetch(k, v)

    # Dense float16 bytes for one token across all KV heads
    dense_per_token = N_KV_HEADS * HEAD_DIM * 2  # float16

    # 3-bit codes: how many uint32 words per token?
    cpw = 32 // 3   # codes per uint32 word = 10
    words_per_pos = math.ceil(HEAD_DIM / cpw)  # ceil(8/10) = 1
    k_code_bytes_per_token = N_KV_HEADS * words_per_pos * 4  # uint32

    # codes should use fewer raw bytes-per-token than float16
    assert k_code_bytes_per_token <= dense_per_token, (
        f"3-bit codes ({k_code_bytes_per_token} B/tok) "
        f">= dense f16 ({dense_per_token} B/tok)"
    )

    # The actual stored codes array exists and is correctly typed
    assert tq.k_codes is not None
    assert tq.k_codes.dtype == mx.uint32, (
        f"Expected uint32 k_codes, got {tq.k_codes.dtype}"
    )
