"""
turboquant.kernels -- fused Metal kernels for hot-path decode operations.

These kernels replace the three-step MLX op chain
  (unpack_codes -> dequantize_groups -> decode_topk_residual)
with a single Metal shader dispatch, eliminating intermediate tensors and
reducing Metal command-encoder round-trips.

Public API
----------
decode_k_fused(packed, scales, resid_vals, resid_idx, *,
               bits, group_size, topk, d_head, out_dtype)
    Single-pass unpack + dequant + residual scatter for K vectors.

decode_v_fused(packed, scales, *, bits, group_size, d_head, out_dtype)
    Single-pass unpack + dequant for V vectors (no residual).

Falls back to pure-MLX ops on environments without Metal
(e.g. public CI runners).
"""
from __future__ import annotations

import math

try:
    import mlx.core as mx
    _HAS_METAL = True
except ImportError:  # pragma: no cover
    _HAS_METAL = False

__all__ = ["decode_k_fused", "decode_v_fused", "_HAS_METAL"]

# ---------------------------------------------------------------------------
# Metal kernel source templates
#
# Compile-time constants are baked in via .format().  The template parameter
# T is the output dtype (half / float / bfloat) resolved by MLX at call time.
# n_tokens is passed as a 1-element uint32 buffer for bounds-checking because
# it varies per call while all other dims are config-constant.
# ---------------------------------------------------------------------------

_SRC_NO_RESIDUAL = """
    uint gid = thread_position_in_grid.x;
    if (gid >= n_tokens_buf[0] * {d_out}u) return;

    constexpr uint D       = {d_out}u;
    constexpr uint BITS    = {bits}u;
    constexpr uint GSIZ    = {group_size}u;
    constexpr uint NWORDS  = {n_words}u;
    constexpr uint NGROUPS = {n_groups}u;
    constexpr uint CPW     = 32u / BITS;
    constexpr uint MASK    = (1u << BITS) - 1u;
    constexpr int  QMAX    = (1 << (int(BITS) - 1)) - 1;

    uint d = gid % D;
    uint t = gid / D;

    uint word_idx = d / CPW;
    uint bit_pos  = (d % CPW) * BITS;
    uint code     = (packed[t * NWORDS + word_idx] >> bit_pos) & MASK;

    uint  g     = d / GSIZ;
    float scale = float(scales[t * NGROUPS + g]);
    float val   = float((int)code - QMAX) * scale;

    out[gid] = T(val);
"""

_SRC_WITH_RESIDUAL = """
    uint gid = thread_position_in_grid.x;
    if (gid >= n_tokens_buf[0] * {d_out}u) return;

    constexpr uint D       = {d_out}u;
    constexpr uint BITS    = {bits}u;
    constexpr uint GSIZ    = {group_size}u;
    constexpr uint NWORDS  = {n_words}u;
    constexpr uint NGROUPS = {n_groups}u;
    constexpr uint TOPK    = {topk}u;
    constexpr uint CPW     = 32u / BITS;
    constexpr uint MASK    = (1u << BITS) - 1u;
    constexpr int  QMAX    = (1 << (int(BITS) - 1)) - 1;

    uint d = gid % D;
    uint t = gid / D;

    uint word_idx = d / CPW;
    uint bit_pos  = (d % CPW) * BITS;
    uint code     = (packed[t * NWORDS + word_idx] >> bit_pos) & MASK;

    uint  g       = d / GSIZ;
    uint  d_in_g  = d % GSIZ;
    float scale   = float(scales[t * NGROUPS + g]);
    float val     = float((int)code - QMAX) * scale;

    float res   = 0.0f;
    uint  rbase = (t * NGROUPS + g) * TOPK;
    for (uint ki = 0u; ki < TOPK; ki++) {{
        if ((uint)resid_idx[rbase + ki] == d_in_g) {{
            res = float(resid_vals[rbase + ki]);
            break;
        }}
    }}

    out[gid] = T(val + res);
"""

# ---------------------------------------------------------------------------
# Kernel cache
# ---------------------------------------------------------------------------

_kernel_cache: dict = {}
_TG = 256  # threadgroup size -- matches Apple GPU SIMD width


def _get_kernel(bits: int, group_size: int, topk: int,
                n_words: int, n_groups: int, d_head: int):
    key = (bits, group_size, topk, n_words, n_groups, d_head)
    if key in _kernel_cache:
        return _kernel_cache[key]

    has_res = topk > 0
    fmt = dict(bits=bits, group_size=group_size, n_words=n_words,
               n_groups=n_groups, d_out=d_head)

    if has_res:
        src = _SRC_WITH_RESIDUAL.format(topk=topk, **fmt)
        kernel = mx.fast.metal_kernel(
            name=f"tq_dk_{bits}b{group_size}g{topk}k{d_head}d",
            input_names=["packed", "scales", "resid_vals",
                         "resid_idx", "n_tokens_buf"],
            output_names=["out"],
            source=src,
        )
    else:
        src = _SRC_NO_RESIDUAL.format(**fmt)
        kernel = mx.fast.metal_kernel(
            name=f"tq_dk_{bits}b{group_size}g{d_head}d",
            input_names=["packed", "scales", "n_tokens_buf"],
            output_names=["out"],
            source=src,
        )

    _kernel_cache[key] = kernel
    return kernel


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def decode_k_fused(
    packed,
    scales,
    resid_vals,
    resid_idx,
    *,
    bits: int,
    group_size: int,
    topk: int,
    d_head: int,
    out_dtype=None,
):
    """Fused unpack + dequant + residual-scatter Metal kernel.

    Parameters
    ----------
    packed      [..., n_words]          uint32
    scales      [..., n_groups]         float16
    resid_vals  [..., n_groups, topk]   float16  |  None
    resid_idx   [..., n_groups, topk]   uint8    |  None
    bits        quantisation bit-width (3 or 4)
    group_size  quantisation group size
    topk        residual top-k (0 = no residual)
    d_head      output head dimension
    out_dtype   output dtype; defaults to scales.dtype

    Returns
    -------
    mx.array  [..., d_head]
    """
    if not _HAS_METAL:  # pragma: no cover
        return _fallback_decode(packed, scales, resid_vals, resid_idx,
                                bits=bits, group_size=group_size,
                                topk=topk, d_head=d_head)

    if out_dtype is None:
        out_dtype = scales.dtype

    *prefix, n_words = packed.shape
    n_groups = scales.shape[-1]
    n_tokens = math.prod(prefix) if prefix else 1

    has_res = topk > 0 and resid_vals is not None and resid_idx is not None
    eff_topk = topk if has_res else 0

    kernel = _get_kernel(bits, group_size, eff_topk, n_words, n_groups, d_head)
    total  = n_tokens * d_head
    grid_x = math.ceil(total / _TG) * _TG

    n_tok_arr = mx.array([n_tokens], dtype=mx.uint32)
    pk_flat   = packed.reshape(n_tokens, n_words)
    sc_flat   = scales.reshape(n_tokens, n_groups)

    if has_res:
        rv_flat = resid_vals.reshape(n_tokens, n_groups * topk)
        ri_flat = resid_idx.reshape(n_tokens, n_groups * topk)
        outputs = kernel(
            inputs=[pk_flat, sc_flat, rv_flat, ri_flat, n_tok_arr],
            template=[("T", out_dtype)],
            grid=(grid_x, 1, 1),
            threadgroup=(_TG, 1, 1),
            output_shapes=[(total,)],
            output_dtypes=[out_dtype],
        )
    else:
        outputs = kernel(
            inputs=[pk_flat, sc_flat, n_tok_arr],
            template=[("T", out_dtype)],
            grid=(grid_x, 1, 1),
            threadgroup=(_TG, 1, 1),
            output_shapes=[(total,)],
            output_dtypes=[out_dtype],
        )

    return outputs[0].reshape(*prefix, d_head) if prefix else outputs[0].reshape(d_head)


def decode_v_fused(packed, scales, *, bits: int, group_size: int,
                   d_head: int, out_dtype=None):
    """Fused unpack + dequant Metal kernel for V vectors (no residual)."""
    return decode_k_fused(
        packed, scales, None, None,
        bits=bits, group_size=group_size, topk=0,
        d_head=d_head, out_dtype=out_dtype,
    )


# ---------------------------------------------------------------------------
# Pure-MLX fallback (non-Apple-Silicon / test environments)
# ---------------------------------------------------------------------------

def _fallback_decode(packed, scales, resid_vals, resid_idx, *,
                     bits, group_size, topk, d_head):
    from turboquant.core.quantizer import dequantize_groups
    from turboquant.core.residual import decode_topk_residual

    d_pad = scales.shape[-1] * group_size
    y_hat = dequantize_groups(packed, scales, bits, group_size, d_pad)

    if topk > 0 and resid_vals is not None:
        residual = decode_topk_residual(resid_vals, resid_idx, group_size)
        y_hat = y_hat + residual[..., :d_pad]

    return y_hat[..., :d_head]
