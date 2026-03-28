"""
mlx_lm.cache_upgrade — production KV-cache upgrade policy.

This module owns the policy for when and how to promote a dense KVCache to a
TurboQuantKCache.  It is intentionally separate from ``generate.py`` so that
the policy can be unit-tested, reused across frontends, and evolved without
touching the main generation loop.

The legacy helper ``maybe_turboquant_k_cache`` in ``generate.py`` now
delegates to :func:`upgrade_cache_list`.

Usage
-----
    from turboquant.config import TurboQuantConfig
    from mlx_lm.cache_upgrade import upgrade_cache_list

    config = TurboQuantConfig(k_bits=3, k_group_size=64, ...)
    events = upgrade_cache_list(prompt_cache, k_start=512, config=config)
    for ev in events:
        if ev.upgraded:
            print(f"layer {ev.layer_index}: {ev.old_type} → {ev.new_type} "
                  f"at offset {ev.offset_at_upgrade}")
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


# ── Event ─────────────────────────────────────────────────────────────────────

@dataclass
class CacheUpgradeEvent:
    """Record of a single cache-layer upgrade decision.

    Fields
    ------
    upgraded:
        ``True`` if the layer was promoted to TurboQuantKCache.
    layer_index:
        Zero-based index of the layer in *prompt_cache*.
    old_type:
        ``type(cache).__name__`` before the upgrade (or the same type if
        no upgrade occurred).
    new_type:
        ``type(cache).__name__`` after the upgrade.
    offset_at_upgrade:
        ``cache.offset`` at the moment the decision was made.
    """
    upgraded: bool
    layer_index: int
    old_type: str
    new_type: str
    offset_at_upgrade: int


# ── Upgrade policy ────────────────────────────────────────────────────────────

def upgrade_cache_list(
    prompt_cache: list,
    k_start: Optional[int],
    config: "TurboQuantConfig",  # noqa: F821 — resolved at runtime
) -> List[CacheUpgradeEvent]:
    """Promote KVCache entries to TurboQuantKCache when their offset threshold
    is reached.

    This is the canonical upgrade path used by the mlx-lm generation loop.
    Call once per generation step; the function is idempotent — layers that
    have already been upgraded are skipped.

    Parameters
    ----------
    prompt_cache:
        The per-layer cache list.  Modified in place when an upgrade occurs.
    k_start:
        Minimum ``cache.offset`` before upgrading.  ``None`` disables all
        upgrades (every layer stays as-is).
    config:
        :class:`turboquant.config.TurboQuantConfig` governing compression.
        The production path always uses ``return_mode="view"``; the legacy
        ``return_mode`` kwarg is not surfaced here.

    Returns
    -------
    List[CacheUpgradeEvent]
        One event per cache layer, in order.  Inspect ``ev.upgraded`` to
        see which layers were promoted this call.
    """
    # Lazy import to avoid circular deps and to keep this module importable
    # even if turboquant or mlx_lm is not fully initialised.
    from mlx_lm.models.cache import TurboQuantKCache

    events: List[CacheUpgradeEvent] = []

    if k_start is None:
        # Fast path: no upgrade policy in effect.
        for i, c in enumerate(prompt_cache):
            events.append(
                CacheUpgradeEvent(
                    upgraded=False,
                    layer_index=i,
                    old_type=type(c).__name__,
                    new_type=type(c).__name__,
                    offset_at_upgrade=getattr(c, "offset", 0),
                )
            )
        return events

    for i, c in enumerate(prompt_cache):
        old_type = type(c).__name__
        cur_offset = getattr(c, "offset", 0)

        # Already upgraded — skip.
        if isinstance(c, TurboQuantKCache):
            events.append(
                CacheUpgradeEvent(
                    upgraded=False,
                    layer_index=i,
                    old_type=old_type,
                    new_type=old_type,
                    offset_at_upgrade=cur_offset,
                )
            )
            continue

        # Threshold not yet reached or no upgrade hook on this cache type.
        if not hasattr(c, "to_turboquant") or cur_offset < k_start:
            events.append(
                CacheUpgradeEvent(
                    upgraded=False,
                    layer_index=i,
                    old_type=old_type,
                    new_type=old_type,
                    offset_at_upgrade=cur_offset,
                )
            )
            continue

        # Upgrade: map production TurboQuantConfig → legacy to_turboquant kwargs.
        # The production path always uses view mode (streaming attention).
        # resid_scale_bits remains adapter metadata only; residual_topk carries
        # the actual sparse-residual behavior into the production compressor.
        prompt_cache[i] = c.to_turboquant(
            main_bits=config.k_bits,
            group_size=config.k_group_size,
            rotation=config.rotation,
            return_mode="view",
            resid_scale_bits=8,
            residual_topk=config.residual_topk,
            v_bits=config.v_bits,
            v_group_size=config.v_group_size,
            v_enabled=config.v_enabled,
            block_tokens=config.block_tokens,
        )
        events.append(
            CacheUpgradeEvent(
                upgraded=True,
                layer_index=i,
                old_type=old_type,
                new_type=type(prompt_cache[i]).__name__,
                offset_at_upgrade=cur_offset,
            )
        )

    return events
