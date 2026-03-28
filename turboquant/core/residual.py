"""
Top-k sparse residual correction.

After the main quantiser, a fraction of the quantisation error can be
captured cheaply by storing the *k* largest-magnitude residual components
per group.  Encoding cost: k × (fp16 value + uint8 index) per group.

For k=2, group_size=64, head_dim=128:
  overhead = 2 groups × 2 × (2 B + 1 B) = 12 B / token / head
  vs old sign-sketch ≈ 5 B — but recovers substantially more signal.

Decode (scatter)
----------------
MLX does not yet expose a stable scatter_nd Metal primitive, so we
reconstruct via broadcast comparison: build a [n_groups, k, g] boolean
mask and reduce.  For typical (k=2, g=64) shapes this intermediate is
small (128 elements per group).

A TODO: replace with mx.fast.metal_kernel scatter once the API stabilises.
"""
from __future__ import annotations

from typing import Tuple

import mlx.core as mx


def encode_topk_residual(
    residual: mx.array,
    k: int,
    group_size: int,
) -> Tuple[mx.array, mx.array]:
    """Compute per-group top-k sparse representation of *residual*.

    Parameters
    ----------
    residual:   [..., d_pad]  — must be divisible by group_size
    k:          number of top-k components to keep per group
    group_size: group width (same as quantiser group_size)

    Returns
    -------
    values:  [..., n_groups, k]  float16  — actual residual values
    indices: [..., n_groups, k]  uint8    — position in [0, group_size)
    """
    *prefix, d_pad = residual.shape
    assert d_pad % group_size == 0, (
        f"d_pad={d_pad} must be divisible by group_size={group_size}"
    )
    n_groups = d_pad // group_size

    rg = residual.reshape(*prefix, n_groups, group_size)  # [..., ng, g]

    # Sort indices by descending absolute magnitude
    abs_rg = mx.abs(rg)                                   # [..., ng, g]
    sort_idx = mx.argsort(-abs_rg, axis=-1)               # [..., ng, g]
    topk_idx = sort_idx[..., :k]                          # [..., ng, k]
    topk_vals = mx.take_along_axis(rg, topk_idx, axis=-1) # [..., ng, k]

    return topk_vals.astype(mx.float16), topk_idx.astype(mx.uint8)


def decode_topk_residual(
    values: mx.array,
    indices: mx.array,
    group_size: int,
) -> mx.array:
    """Reconstruct sparse residual from (values, indices).

    Parameters
    ----------
    values:     [..., n_groups, k]  float (any)
    indices:    [..., n_groups, k]  uint8 / int
    group_size: must match encoding group_size

    Returns
    -------
    residual: [..., n_groups * group_size]  same dtype as values
    """
    *prefix, n_groups, k = values.shape
    g = group_size

    # Build position range [g] for scatter-via-comparison
    col = mx.arange(g, dtype=indices.dtype)   # [g]

    # indices[..., None] : [..., ng, k, 1]
    # col               : [g]  (broadcasts to [..., ng, k, g])
    mask = (indices[..., None] == col)         # [..., ng, k, g]  bool

    # values[..., None]  : [..., ng, k, 1]
    # mask               : [..., ng, k, g]
    # product → sum over k → [..., ng, g]
    out = mx.sum(
        values[..., None].astype(mx.float32) * mask.astype(mx.float32),
        axis=-2,
    ).astype(values.dtype)                     # [..., ng, g]

    return out.reshape(*prefix, n_groups * g)  # [..., d_pad]
