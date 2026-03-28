"""
turboquant.eval.memory
~~~~~~~~~~~~~~~~~~~~~~

Peak-memory profiling helpers for comparing dense vs TurboQuant KV caches.

These utilities are intentionally simple: they measure the live object graph
of MLX arrays rather than system-level memory (which requires platform APIs).
Use them for relative comparisons between cache types, not absolute figures.

Typical usage
-------------
::

    from turboquant.eval.memory import memory_report

    report = memory_report(
        model=my_model,
        input_ids=ids,
        turboquant_config=TurboQuantConfig(main_bits=3, group_size=64),
    )
    print(report)
    # {'dense_cache_bytes': 2097152, 'tq_cache_bytes': 524288,
    #   'ratio': 4.0, 'n_layers': 18}
"""
from __future__ import annotations

from typing import List

import mlx.core as mx
import mlx.nn as nn


def peak_memory_bytes(cache_list) -> int:
    """Sum the byte sizes of all MLX arrays in a cache list.

    Iterates over each cache object and looks for known array attributes
    (``keys``, ``values``, ``k_codes``, ``k_scales``, ``v_codes``, ``v_scales``).

    Parameters
    ----------
    cache_list:
        A list of cache objects (KVCache or TurboQuantKCache instances).

    Returns
    -------
    int
        Total bytes consumed by tracked arrays.
    """
    total = 0
    for entry in cache_list:
        if entry is None:
            continue
        # TurboQuantKCache exposes nbytes directly
        if hasattr(entry, "nbytes"):
            total += int(entry.nbytes)
        # KVCache stores raw arrays
        elif hasattr(entry, "keys") and entry.keys is not None:
            total += _array_bytes(entry.keys)
            if hasattr(entry, "values") and entry.values is not None:
                total += _array_bytes(entry.values)
    return total


def _array_bytes(a: mx.array) -> int:
    """Return byte size of a single MLX array."""
    itemsize = {
        mx.float16: 2, mx.bfloat16: 2, mx.float32: 4,
        mx.int8: 1, mx.uint8: 1, mx.int16: 2, mx.uint16: 2,
        mx.int32: 4, mx.uint32: 4,
    }.get(a.dtype, 4)
    size = 1
    for d in a.shape:
        size *= d
    return size * itemsize


def memory_report(
    model: nn.Module,
    input_ids: mx.array,
    turboquant_config=None,
    k_start: int = 0,
) -> dict:
    """Compare dense vs TurboQuant cache memory after a single forward pass.

    Parameters
    ----------
    model:
        MLX language model.
    input_ids:
        Shape ``[1, T]`` integer token ids.
    turboquant_config:
        :class:`mlx_lm.models.cache.TurboQuantConfig` or ``None`` (skips TQ run).
    k_start:
        Token index for cache upgrade.

    Returns
    -------
    dict with keys:
        ``dense_cache_bytes``, ``tq_cache_bytes`` (or ``None``),
        ``ratio`` (dense / tq), ``n_layers``
    """
    from mlx_lm.models.cache import make_prompt_cache

    # dense
    dense_cache = make_prompt_cache(model)
    model(input_ids, cache=dense_cache)
    mx.eval(*[c.keys for c in dense_cache if hasattr(c, "keys") and c.keys is not None])
    dense_bytes = peak_memory_bytes(dense_cache)

    tq_bytes: int | None = None
    if turboquant_config is not None:
        from mlx_lm.cache_upgrade import upgrade_cache_list

        tq_cache = make_prompt_cache(model)
        upgrade_cache_list(tq_cache, k_start=k_start, config=turboquant_config)
        model(input_ids, cache=tq_cache)
        tq_arrs = [
            c.k_codes for c in tq_cache
            if hasattr(c, "k_codes") and c.k_codes is not None
        ]
        if tq_arrs:
            mx.eval(*tq_arrs)
        tq_bytes = peak_memory_bytes(tq_cache)

    ratio = (dense_bytes / tq_bytes) if (tq_bytes and tq_bytes > 0) else None

    return {
        "dense_cache_bytes": dense_bytes,
        "tq_cache_bytes":    tq_bytes,
        "ratio":             round(ratio, 2) if ratio is not None else None,
        "n_layers":          len(dense_cache),
    }
