# Copyright © 2023-2024 Apple Inc.

from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .cache import TurboQuantKeysView


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    head_dim: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    rope_theta: float = 10000
    rope_traditional: bool = False


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


def _expand_kv_heads(x: mx.array, target_heads: int) -> mx.array:
    """Broadcast KV heads to match query heads for grouped-query attention."""
    h = x.shape[1]
    if h == target_heads:
        return x
    if target_heads % h != 0:
        raise ValueError(
            f"Cannot expand {h} KV heads to {target_heads} query heads"
        )
    repeats = target_heads // h
    return mx.repeat(x, repeats, axis=1)


def _streaming_softmax_attention(
    q_rot: mx.array,
    keys_view: TurboQuantKeysView,
    *,
    scale: float,
) -> mx.array:
    """Numerically stable online-softmax causal attention over TurboQuant blocks.

    q_rot: [B, H_q, L_q, D]  — already rotated queries
    Returns: [B, H_q, L_q, Dv]
    """
    tq = keys_view.cache
    B, H_q, L_q, _ = q_rot.shape

    q_end   = keys_view.end
    q_start = q_end - L_q
    # q_pos shape: [1, 1, L_q, 1] for causal mask broadcast
    q_pos = mx.arange(q_start, q_end, dtype=mx.int32).reshape(1, 1, L_q, 1)

    m   = mx.full((B, H_q, L_q, 1), -1e30, dtype=mx.float32)
    lse = mx.zeros((B, H_q, L_q, 1), dtype=mx.float32)
    acc = None

    for s, e, k_rot_blk, v_blk in tq.iter_rotated_kv_blocks(keys_view):
        k_rot_blk = _expand_kv_heads(k_rot_blk, H_q)
        v_blk     = _expand_kv_heads(v_blk,     H_q)

        qf = q_rot.astype(mx.float32)
        kf = k_rot_blk.astype(mx.float32)
        vf = v_blk.astype(mx.float32)

        # [B, H_q, L_q, blk]
        scores = mx.matmul(qf, kf.transpose(0, 1, 3, 2)) * scale

        # Causal mask: key positions must be <= query positions
        k_pos  = mx.arange(s, e, dtype=mx.int32).reshape(1, 1, 1, e - s)
        scores = mx.where(k_pos <= q_pos, scores, mx.array(-1e30, dtype=scores.dtype))

        blk_m = mx.max(scores, axis=-1, keepdims=True)
        new_m = mx.maximum(m, blk_m)

        alpha = mx.exp(m - new_m)          # rescale old accumulator
        p     = mx.exp(scores - new_m)    # softmax numerator for this block

        if acc is None:
            Dv  = vf.shape[-1]
            acc = mx.zeros((B, H_q, L_q, Dv), dtype=mx.float32)

        lse = lse * alpha + mx.sum(p, axis=-1, keepdims=True)
        acc = acc * alpha + mx.matmul(p, vf)
        m   = new_m

    if acc is None:
        # No tokens in view — return zeros
        Dv  = q_rot.shape[-1]
        acc = mx.zeros((B, H_q, L_q, Dv), dtype=mx.float32)
        lse = mx.ones((B, H_q, L_q, 1),   dtype=mx.float32)

    return acc / mx.maximum(lse, mx.array(1e-6, dtype=lse.dtype))


def turboquant_streaming_attention(
    queries: mx.array,
    keys_view: TurboQuantKeysView,
    *,
    scale: float,
) -> mx.array:
    """Public entry point: rotate queries then run streaming causal attention."""
    tq    = keys_view.cache
    q_rot = tq.rotate_queries_for_attention(queries)
    return _streaming_softmax_attention(q_rot, keys_view, scale=scale).astype(
        queries.dtype
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim

        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.rope = nn.RoPE(
            head_dim,
            traditional=args.rope_traditional,
            base=args.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        if isinstance(keys, TurboQuantKeysView):
            output = turboquant_streaming_attention(
                queries,
                keys,
                scale=self.scale,
            )
        else:
            output = scaled_dot_product_attention(
                queries, keys, values, cache=cache, scale=self.scale, mask=mask
            )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        return self.down_proj(nn.gelu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.args = args

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class GemmaModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            TransformerBlock(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        h = self.embed_tokens(inputs)
        h = h * (self.args.hidden_size**0.5)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model_type = args.model_type
        self.model = GemmaModel(args)
        self.args = args

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out = self.model(inputs, cache)
        out = self.model.embed_tokens.as_linear(out)
        return out

    @property
    def layers(self):
        return self.model.layers
