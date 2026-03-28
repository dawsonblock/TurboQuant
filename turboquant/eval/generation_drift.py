"""
turboquant.eval.generation_drift
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Measure the KL-divergence between the token probability distributions
produced by a dense KVCache model and a TurboQuant-cached model.

High KL divergence across a sequence indicates that quantisation noise is
meaningfully changing the model's beliefs — useful for identifying problem
layers or bit-width thresholds.

Typical usage
-------------
::

    from turboquant.eval.generation_drift import drift_report

    report = drift_report(
        model=my_model,
        input_ids=ids,          # [1, T]
        turboquant_config=TurboQuantConfig(main_bits=3, group_size=64),
    )
    print(report)
    # {'mean_kl': 0.004, 'max_kl': 0.021, 'n_tokens': 63}
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def logit_kl_divergence(
    logits_p: mx.array,
    logits_q: mx.array,
    temperature: float = 1.0,
) -> mx.array:
    """Per-position KL( softmax(p) || softmax(q) ) averaged over vocab.

    Parameters
    ----------
    logits_p:
        Shape ``[T, V]`` — reference (dense) logits.
    logits_q:
        Shape ``[T, V]`` — approximate (TurboQuant) logits.
    temperature:
        Softmax temperature (default 1.0).

    Returns
    -------
    mx.array
        Shape ``[T]`` — per-position KL divergence.
    """
    if temperature != 1.0:
        logits_p = logits_p / temperature
        logits_q = logits_q / temperature

    log_p = logits_p - mx.logsumexp(logits_p, axis=-1, keepdims=True)
    log_q = logits_q - mx.logsumexp(logits_q, axis=-1, keepdims=True)
    p = mx.exp(log_p)
    # KL(P||Q) = sum_v P(v) * (log P(v) - log Q(v))
    kl = mx.sum(p * (log_p - log_q), axis=-1)  # [T]
    return kl


def _collect_logits(model: nn.Module, input_ids: mx.array, cache) -> mx.array:
    logits = model(input_ids, cache=cache)  # [1, T, V]
    return logits[0]                        # [T, V]


def drift_report(
    model: nn.Module,
    input_ids: mx.array,
    turboquant_config=None,
    k_start: int = 0,
    temperature: float = 1.0,
) -> dict:
    """Compute per-token KL divergence between dense and TQ model outputs.

    Parameters
    ----------
    model:
        MLX language model.
    input_ids:
        Shape ``[1, T]`` integer token ids.
    turboquant_config:
        :class:`mlx_lm.models.cache.TurboQuantConfig` or ``None`` (returns
        zeroed drift if None — useful as a baseline sanity check).
    k_start:
        Token index at which TurboQuant cache upgrade is applied.
    temperature:
        Softmax temperature for the KL computation.

    Returns
    -------
    dict with keys:
        ``mean_kl``, ``max_kl``, ``min_kl``, ``n_tokens``
        and ``kl_per_token`` (list of floats).
    """
    from mlx_lm.models.cache import make_prompt_cache

    feed = input_ids[:, :-1]   # [1, T-1]

    # dense reference
    dense_cache  = make_prompt_cache(model)
    dense_logits = _collect_logits(model, feed, cache=dense_cache)
    mx.eval(dense_logits)

    if turboquant_config is None:
        T = dense_logits.shape[0]
        return {
            "mean_kl":      0.0,
            "max_kl":       0.0,
            "min_kl":       0.0,
            "n_tokens":     T,
            "kl_per_token": [0.0] * T,
        }

    from mlx_lm.cache_upgrade import upgrade_cache_list

    tq_cache = make_prompt_cache(model)
    upgrade_cache_list(tq_cache, k_start=k_start, config=turboquant_config)
    tq_logits = _collect_logits(model, feed, cache=tq_cache)
    mx.eval(tq_logits)

    kl_vec = logit_kl_divergence(dense_logits, tq_logits, temperature=temperature)
    mx.eval(kl_vec)

    kl_list = kl_vec.tolist()
    return {
        "mean_kl":      round(float(mx.mean(kl_vec).item()), 6),
        "max_kl":       round(float(mx.max(kl_vec).item()),  6),
        "min_kl":       round(float(mx.min(kl_vec).item()),  6),
        "n_tokens":     len(kl_list),
        "kl_per_token": [round(x, 6) for x in kl_list],
    }
