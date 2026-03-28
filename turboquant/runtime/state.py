"""
TurboQuant state serialisation schema.

STATE_SCHEMA_VERSION is bumped whenever the dict layout produced by
``KVCompressor.state()`` changes in a backward-incompatible way.

Consumers (save/load, test fixtures, mlx-lm cache migration) must pass
the state dict through ``validate_state`` before restoring a compressor.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from turboquant.config import TurboQuantConfig

# ── Version ───────────────────────────────────────────────────────────────────

STATE_SCHEMA_VERSION: int = 1
"""Integer version of the KVCompressor state dict format.

Changelog
---------
1  (initial)  – keys: schema_version, offset, d_head, d_pad, v_dim, v_pad,
                       k_packed, k_scales, resid_vals, resid_idx,
                       v_packed, v_scales
"""

# Keys that are always present even when the cache is empty.
_REQUIRED_SCALAR_KEYS = frozenset(
    {"schema_version", "offset", "d_head", "d_pad", "v_dim", "v_pad"}
)

# Array keys that are present once the first token has been stored.
_ARRAY_KEYS = frozenset(
    {"k_packed", "k_scales", "resid_vals", "resid_idx", "v_packed", "v_scales"}
)


# ── Public API ────────────────────────────────────────────────────────────────

def validate_state(
    state: Dict[str, Any],
    config: Optional[TurboQuantConfig] = None,
) -> None:
    """Raise ``ValueError`` if *state* is not a valid KVCompressor state dict.

    Parameters
    ----------
    state:
        Dict returned by ``KVCompressor.state()``.
    config:
        If provided, shape consistency between *state* and the config
        (group sizes, bits) is checked in addition to schema validation.

    Raises
    ------
    ValueError
        On any schema or consistency violation.  The message describes the
        first detected problem.
    """
    # ── version ───────────────────────────────────────────────────────────────
    if "schema_version" not in state:
        raise ValueError(
            "State dict is missing 'schema_version'. "
            "This state was produced by an older KVCompressor (pre-v1). "
            "Re-run prefill to rebuild the cache."
        )

    version = state["schema_version"]
    if not isinstance(version, int):
        raise ValueError(
            f"'schema_version' must be an int, got {type(version).__name__!r}."
        )
    if version != STATE_SCHEMA_VERSION:
        raise ValueError(
            f"State schema version {version} is incompatible with the "
            f"current loader (expects {STATE_SCHEMA_VERSION}). "
            "Re-run prefill to rebuild the cache."
        )

    # ── required scalars ──────────────────────────────────────────────────────
    missing = _REQUIRED_SCALAR_KEYS - state.keys()
    if missing:
        raise ValueError(
            f"State dict is missing required keys: {sorted(missing)}."
        )

    offset = state["offset"]
    if not isinstance(offset, int) or offset < 0:
        raise ValueError(
            f"'offset' must be a non-negative int, got {offset!r}."
        )

    # ── array presence consistent with offset ────────────────────────────────
    if offset > 0:
        k_packed = state.get("k_packed")
        if k_packed is None:
            raise ValueError(
                f"State has offset={offset} but 'k_packed' is None. "
                "State is corrupt."
            )
        # Token axis is axis-2 for all compressed arrays [B, H, T, ...]
        if hasattr(k_packed, "shape") and k_packed.shape[2] < offset:
            raise ValueError(
                f"'k_packed' token dimension ({k_packed.shape[2]}) is "
                f"smaller than offset ({offset}). State is corrupt."
            )

    # ── config-level shape checks (optional) ──────────────────────────────────
    if config is not None and offset > 0:
        k_packed = state.get("k_packed")
        k_scales = state.get("k_scales")

        if k_packed is not None and k_scales is not None:
            if hasattr(k_scales, "shape"):
                ng_stored = k_scales.shape[-1]
                d_pad = state.get("d_pad")
                if d_pad is not None:
                    ng_expected = d_pad // config.k_group_size
                    if ng_stored != ng_expected:
                        raise ValueError(
                            f"k_scales group count {ng_stored} does not match "
                            f"config.k_group_size={config.k_group_size} with "
                            f"d_pad={d_pad} (expected {ng_expected} groups)."
                        )
