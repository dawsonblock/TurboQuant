"""
Layout enforcement for TurboQuant tensors.

All tensors entering the compression pipeline must be [B, H, T, D].
``ensure_layout`` validates the shape and returns the tensor unchanged
(MLX arrays are always contiguous; no copy is needed).
"""
from __future__ import annotations

import mlx.core as mx


def ensure_layout(x: mx.array, name: str = "tensor") -> mx.array:
    """Assert x has shape [B, H, T, D] and return it unchanged.

    Parameters
    ----------
    x:    Input array.
    name: Label used in the error message (default: "tensor").

    Raises
    ------
    ValueError if ndim != 4 or any dimension is 0.
    """
    if x.ndim != 4:
        raise ValueError(
            f"TurboQuant requires [B, H, T, D] (4-D) tensors. "
            f"Got {name}.shape = {x.shape} (ndim={x.ndim})."
        )
    B, H, T, D = x.shape
    if B == 0 or H == 0 or T == 0 or D == 0:
        raise ValueError(
            f"All dimensions must be > 0; got {name}.shape = {x.shape}."
        )
    return x
