"""
turboquant.eval.perplexity
~~~~~~~~~~~~~~~~~~~~~~~~~~

Compute token-level perplexity from a sequence of logit tensors or
directly from a running model with a TurboQuantKCache.

Typical usage
-------------
::

    from turboquant.eval.perplexity import perplexity_report

    report = perplexity_report(
        model=my_model,
        tokenizer=my_tokenizer,
        prompt="The quick brown fox",
        turboquant_config=TurboQuantConfig(main_bits=3, group_size=64),
    )
    print(report)
    # {'dense_ppl': 12.3, 'tq_ppl': 12.6, 'delta_ppl': 0.3, 'n_tokens': 8}
"""
from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


def perplexity_from_logits(
    logits: mx.array,
    targets: mx.array,
) -> float:
    """Compute perplexity from a [T, V] logit array and [T] target-id array.

    Parameters
    ----------
    logits:
        Shape ``[T, vocab_size]``, raw (pre-softmax) logits.
    targets:
        Shape ``[T]``, integer token ids (ground-truth next tokens).

    Returns
    -------
    float
        Perplexity = exp(mean NLL).
    """
    if logits.shape[0] == 0:
        return float("nan")

    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    T = targets.shape[0]
    nll_sum = -float(mx.sum(log_probs[mx.arange(T), targets]).item())
    return math.exp(nll_sum / T)


def _collect_logits(model: nn.Module, input_ids: mx.array, cache) -> mx.array:
    """Run the model for one forward pass and return [T, V] logits.

    ``input_ids`` is shape ``[1, T]``.  Returns ``[T, vocab_size]``.
    """
    logits = model(input_ids, cache=cache)  # [1, T, V]
    return logits[0]  # [T, V]


def perplexity_report(
    model: nn.Module,
    input_ids: mx.array,
    turboquant_config=None,
    k_start: int = 0,
) -> dict:
    """Measure dense vs TurboQuant perplexity on a tokenised sequence.

    Parameters
    ----------
    model:
        An MLX language model with a ``__call__(input_ids, cache=...)`` API.
    input_ids:
        Shape ``[1, T]`` integer token ids.
    turboquant_config:
        A :class:`mlx_lm.models.cache.TurboQuantConfig` instance, or ``None``
        to skip the TurboQuant run and return only ``dense_ppl``.
    k_start:
        Token index at which to start the TurboQuant cache upgrade (if using
        :func:`mlx_lm.cache_upgrade.upgrade_cache_list`).

    Returns
    -------
    dict with keys:
        ``dense_ppl``, ``tq_ppl`` (or ``None``), ``delta_ppl``, ``n_tokens``
    """
    from mlx_lm.models.cache import make_prompt_cache

    targets = input_ids[0, 1:]      # [T-1]  (next-token targets)
    feed    = input_ids[:, :-1]     # [1, T-1]

    # ---- dense ----
    dense_cache = make_prompt_cache(model)
    dense_logits = _collect_logits(model, feed, cache=dense_cache)  # [T-1, V]
    mx.eval(dense_logits)
    dense_ppl = perplexity_from_logits(dense_logits, targets)

    tq_ppl: Optional[float] = None
    if turboquant_config is not None:
        from mlx_lm.models.cache import make_prompt_cache, TurboQuantKCache
        from mlx_lm.cache_upgrade import upgrade_cache_list

        tq_cache = make_prompt_cache(model)
        upgrade_cache_list(tq_cache, k_start=k_start, config=turboquant_config)
        tq_logits = _collect_logits(model, feed, cache=tq_cache)
        mx.eval(tq_logits)
        tq_ppl = perplexity_from_logits(tq_logits, targets)

    delta = (tq_ppl - dense_ppl) if tq_ppl is not None else None
    return {
        "dense_ppl":  round(dense_ppl, 4),
        "tq_ppl":     round(tq_ppl, 4)  if tq_ppl is not None else None,
        "delta_ppl":  round(delta,   4)  if delta   is not None else None,
        "n_tokens":   int(targets.shape[0]),
    }
