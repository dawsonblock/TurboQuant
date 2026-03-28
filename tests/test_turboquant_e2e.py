import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.cache import KVCache, TurboQuantKCache
from mlx_lm.generate import generate_step, maybe_turboquant_k_cache


class _TinyLM(nn.Module):
    """
    Minimal model stub for generate_step smoke testing.

    Mimics the generate_step contract:
    - input tokens: [B, T]   (generate_step adds the batch dim via [None])
    - calls cache.update_and_fetch so KVCache.state works after prefill
    - returns logits: [B, T, vocab]
    """

    def __init__(self, vocab_size=32, n_kv_heads=2, head_dim=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

    def __call__(self, x, cache=None):
        b, t = x.shape

        # Populate each layer cache so KVCache.state / state roundtrip works
        if cache is not None:
            dummy_k = mx.zeros((b, self.n_kv_heads, t, self.head_dim))
            dummy_v = mx.zeros((b, self.n_kv_heads, t, self.head_dim))
            for c in cache:
                c.update_and_fetch(dummy_k, dummy_v)

        # Bias toward token 1 so argmax sampling is deterministic.
        # Construct via concatenation to avoid MLX in-place issues.
        before = mx.zeros((b, t, 1), dtype=mx.float32)
        one    = mx.ones( (b, t, 1), dtype=mx.float32)
        after  = mx.zeros((b, t, self.vocab_size - 2), dtype=mx.float32)
        return mx.concatenate([before, one, after], axis=-1)


def _argmax_sampler(logits):
    return mx.argmax(logits, axis=-1)


def _make_prompt_cache(offset=0):
    c = KVCache()
    if offset > 0:
        keys = mx.zeros((1, 2, offset, 8), dtype=mx.float32)
        values = mx.zeros((1, 2, offset, 8), dtype=mx.float32)
        c.update_and_fetch(keys, values)
    return [c]


def test_generate_step_smoke_without_turboquant():
    model = _TinyLM(vocab_size=16)
    prompt = mx.array([2, 3, 4], dtype=mx.int32)   # 1-D: generate_step slices [seq]
    prompt_cache = _make_prompt_cache(offset=0)

    gen = generate_step(
        prompt,
        model,
        max_tokens=2,
        sampler=_argmax_sampler,
        prompt_cache=prompt_cache,
        prefill_step_size=4,
    )

    tok1, _ = next(gen)
    tok2, _ = next(gen)

    assert isinstance(tok1, int), f"Expected int token, got {type(tok1)}"
    assert isinstance(tok2, int), f"Expected int token, got {type(tok2)}"
    assert isinstance(prompt_cache[0], KVCache)


def test_generate_step_upgrades_cache_to_turboquant_after_threshold():
    model = _TinyLM(vocab_size=16)
    prompt = mx.array([2, 3, 4, 5], dtype=mx.int32)  # 1-D
    prompt_cache = _make_prompt_cache(offset=0)

    gen = generate_step(
        prompt,
        model,
        max_tokens=2,
        sampler=_argmax_sampler,
        prompt_cache=prompt_cache,
        prefill_step_size=4,
        turboquant_k_start=1,
        turboquant_main_bits=3,
        turboquant_group_size=8,
        turboquant_rotation="identity",
        turboquant_return_mode="view",
        turboquant_resid_scale_bits=8,
        turboquant_v_bits=4,
        turboquant_v_group_size=8,
        turboquant_v_enabled=True,
    )

    tok1, _ = next(gen)

    assert isinstance(tok1, int), f"Expected int token, got {type(tok1)}"
    assert isinstance(prompt_cache[0], TurboQuantKCache)
    assert prompt_cache[0].offset >= len(prompt)


def test_generate_step_turboquant_cache_keeps_growing():
    model = _TinyLM(vocab_size=16)
    prompt = mx.array([7, 8, 9], dtype=mx.int32)  # 1-D
    prompt_cache = _make_prompt_cache(offset=0)

    gen = generate_step(
        prompt,
        model,
        max_tokens=3,
        sampler=_argmax_sampler,
        prompt_cache=prompt_cache,
        prefill_step_size=4,
        turboquant_k_start=1,
        turboquant_main_bits=3,
        turboquant_group_size=8,
        turboquant_rotation="identity",
        turboquant_return_mode="view",
        turboquant_resid_scale_bits=8,
        turboquant_v_bits=4,
        turboquant_v_group_size=8,
        turboquant_v_enabled=True,
    )

    _ = next(gen)
    first_offset = prompt_cache[0].offset

    _ = next(gen)
    second_offset = prompt_cache[0].offset

    assert isinstance(prompt_cache[0], TurboQuantKCache)
    assert second_offset >= first_offset


def test_maybe_turboquant_k_cache_direct_hook_smoke():
    prompt_cache = _make_prompt_cache(offset=4)

    maybe_turboquant_k_cache(
        prompt_cache,
        turboquant_k_start=4,
        turboquant_main_bits=3,
        turboquant_group_size=8,
        turboquant_rotation="identity",
        turboquant_return_mode="view",
        turboquant_resid_scale_bits=8,
        turboquant_v_bits=4,
        turboquant_v_group_size=8,
        turboquant_v_enabled=True,
    )

    assert isinstance(prompt_cache[0], TurboQuantKCache)
    assert prompt_cache[0].offset == 4
