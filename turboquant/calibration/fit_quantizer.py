"""
calibrate() — offline per-layer quantiser fitting.

Typical usage::

    pipeline = TurboQuantPipeline(config, layer_id=0)
    calibrate(pipeline, kv_samples, mode="k")
    # pipeline.pipeline._k_quant is now fitted

The calibration loop collects KV tensors from a data-loader-style
iterator, concatenates them along the sequence axis, and calls
``GroupScalarQuantizer.fit`` for K and/or V.

``extract_kv`` is a user-supplied function that receives each batch
and returns (keys, values) as MLX arrays in [B, H, T, D] layout.
"""
from __future__ import annotations

from typing import Callable, Iterable, Literal, Optional, Tuple

import mlx.core as mx

from turboquant.core.pipeline import TurboQuantPipeline
from turboquant.runtime.layout import ensure_layout


def calibrate(
    pipeline: TurboQuantPipeline,
    data_iter: Iterable,
    extract_kv: Callable[
        [object], Tuple[Optional[mx.array], Optional[mx.array]]
    ],
    mode: Literal["k", "v", "both"] = "both",
    max_batches: int = 64,
) -> TurboQuantPipeline:
    """Calibrate quantiser scale statistics from representative data.

    Parameters
    ----------
    pipeline:    ``TurboQuantPipeline`` to calibrate in-place.
    data_iter:   Any iterable yielding batches.
    extract_kv:  ``fn(batch) -> (keys | None, values | None)``
                 Return ``None`` for a tensor you do not want to collect.
    mode:        Which path to calibrate: "k", "v", or "both".
    max_batches: Upper bound on batches consumed from *data_iter*.

    Returns
    -------
    The same ``pipeline`` (mutated in-place).
    """
    k_samples: list = []
    v_samples: list = []

    for i, batch in enumerate(data_iter):
        if i >= max_batches:
            break
        keys, values = extract_kv(batch)

        if mode in ("k", "both") and keys is not None:
            keys = ensure_layout(keys, "keys")
            B, H, T, D = keys.shape
            # Rotate before collecting so scales match the encoded domain
            rot_k = pipeline.rotate_queries(
                keys.reshape(B * H, T, D)
            )
            k_samples.append(rot_k.reshape(B * H * T, D))
            mx.eval(rot_k)

        if mode in ("v", "both") and values is not None:
            values = ensure_layout(values, "values")
            B, H, T, D = values.shape
            v_samples.append(values.reshape(B * H * T, D))
            mx.eval(values)

    if mode in ("k", "both") and k_samples:
        k_data = mx.concatenate(k_samples, axis=0)   # [N, D]
        mx.eval(k_data)
        pipeline.fit_k(k_data)

    if mode in ("v", "both") and v_samples:
        v_data = mx.concatenate(v_samples, axis=0)   # [N, D]
        mx.eval(v_data)
        pipeline.fit_v(v_data)

    return pipeline
