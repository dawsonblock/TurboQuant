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

STATE_SCHEMA_VERSION: int = 2
"""Integer version of the KVCompressor state dict format.

Changelog
---------
1  initial  – keys: schema_version, offset, d_head, d_pad, v_dim, v_pad,
              k_packed, k_scales, resid_vals, resid_idx, v_packed, v_scales
2  current  – adds config fingerprint and optional calibrated quantizer scales
              so restores can fail closed on config drift and preserve future
              encode behavior after restore.
"""

_SUPPORTED_VERSIONS = frozenset({1, 2})
_REQUIRED_SCALAR_KEYS = frozenset(
    {"schema_version", "offset", "d_head", "d_pad", "v_dim", "v_pad"}
)
_CONFIG_KEYS_V2 = frozenset(
    {
        "k_bits", "k_group_size", "v_bits", "v_group_size", "v_enabled",
        "rotation", "rotation_seed", "residual_topk", "scale_dtype",
        "v_scale_dtype", "eps",
    }
)


def _shape_token_len(arr) -> Optional[int]:
    if arr is None or not hasattr(arr, "shape"):
        return None
    if len(arr.shape) < 3:
        return None
    return arr.shape[2]


def _expect_config_match(state: Dict[str, Any], config: TurboQuantConfig) -> None:
    mismatches = []
    checks = {
        "k_bits": config.k_bits,
        "k_group_size": config.k_group_size,
        "v_bits": config.v_bits,
        "v_group_size": config.v_group_size,
        "v_enabled": config.v_enabled,
        "rotation": config.rotation,
        "rotation_seed": config.rotation_seed,
        "residual_topk": config.residual_topk,
        "scale_dtype": config.scale_dtype,
        "v_scale_dtype": config.v_scale_dtype,
    }
    for key, expected in checks.items():
        if key in state and state[key] != expected:
            mismatches.append(f"{key}: state={state[key]!r}, config={expected!r}")

    if "eps" in state and abs(float(state["eps"]) - float(config.eps)) > 1e-12:
        mismatches.append(f"eps: state={state['eps']!r}, config={config.eps!r}")

    if mismatches:
        raise ValueError(
            "State/config mismatch detected. Refusing restore because future "
            "encode behavior would diverge: " + "; ".join(mismatches)
        )


def validate_state(
    state: Dict[str, Any],
    config: Optional[TurboQuantConfig] = None,
) -> None:
    """Raise ``ValueError`` if *state* is not a valid KVCompressor state dict."""
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
    if version not in _SUPPORTED_VERSIONS:
        raise ValueError(
            f"State schema version {version} is incompatible with the "
            f"current loader (supports {sorted(_SUPPORTED_VERSIONS)}). "
            "Re-run prefill to rebuild the cache."
        )

    missing = _REQUIRED_SCALAR_KEYS - state.keys()
    if missing:
        raise ValueError(f"State dict is missing required keys: {sorted(missing)}.")

    offset = state["offset"]
    if not isinstance(offset, int) or offset < 0:
        raise ValueError(f"'offset' must be a non-negative int, got {offset!r}.")

    if offset > 0:
        k_packed = state.get("k_packed")
        if k_packed is None:
            raise ValueError(
                f"State has offset={offset} but 'k_packed' is None. State is corrupt."
            )
        token_len = _shape_token_len(k_packed)
        if token_len is not None and token_len < offset:
            raise ValueError(
                f"'k_packed' token dimension ({token_len}) is smaller than "
                f"offset ({offset}). State is corrupt."
            )

    if version >= 2:
        missing_cfg = _CONFIG_KEYS_V2 - state.keys()
        if missing_cfg:
            raise ValueError(
                f"State schema v2 is missing config keys: {sorted(missing_cfg)}."
            )

    if config is None:
        return

    if version >= 2:
        _expect_config_match(state, config)

    if offset == 0:
        return

    k_scales = state.get("k_scales")
    d_pad = state.get("d_pad")
    if k_scales is not None and hasattr(k_scales, "shape") and d_pad is not None:
        ng_stored = k_scales.shape[-1]
        ng_expected = d_pad // config.k_group_size
        if ng_stored != ng_expected:
            raise ValueError(
                f"k_scales group count {ng_stored} does not match "
                f"config.k_group_size={config.k_group_size} with d_pad={d_pad} "
                f"(expected {ng_expected} groups)."
            )

    v_scales = state.get("v_scales")
    v_pad = state.get("v_pad")
    if config.v_enabled and v_scales is not None and hasattr(v_scales, "shape") and v_pad is not None:
        vg_stored = v_scales.shape[-1]
        vg_expected = v_pad // config.v_group_size
        if vg_stored != vg_expected:
            raise ValueError(
                f"v_scales group count {vg_stored} does not match "
                f"config.v_group_size={config.v_group_size} with v_pad={v_pad} "
                f"(expected {vg_expected} groups)."
            )

    if version >= 2:
        k_cal = state.get("k_calibrated_scales")
        if k_cal is not None and hasattr(k_cal, "shape") and d_pad is not None:
            expected = d_pad // config.k_group_size
            if k_cal.shape[-1] != expected:
                raise ValueError(
                    f"k_calibrated_scales width {k_cal.shape[-1]} does not match "
                    f"expected group count {expected}."
                )
        v_cal = state.get("v_calibrated_scales")
        if v_cal is not None and hasattr(v_cal, "shape") and v_pad is not None:
            expected = v_pad // config.v_group_size
            if v_cal.shape[-1] != expected:
                raise ValueError(
                    f"v_calibrated_scales width {v_cal.shape[-1]} does not match "
                    f"expected group count {expected}."
                )
