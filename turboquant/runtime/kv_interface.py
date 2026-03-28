"""
KVCompressor — production KV-cache with TurboQuant compression.

Drop-in replacement for a standard KV cache in mlx-lm models that have
been patched for streaming attention (e.g. Gemma).

Interface
---------
    cache = KVCompressor(config, layer_id=0)
    view, v_cur = cache.update_and_fetch(keys, values)
    # -- attention side --
    q_rot = cache.rotate_queries(queries)
    for s, e, k_blk, v_blk in cache.iter_rotated_kv_blocks(view):
        ...  # streaming softmax attention

Single execution path
---------------------
* ``update_and_fetch`` always returns a ``TurboQuantKeysView`` (never a
  dense K tensor).  Models MUST use streaming attention.
* No ``return_mode`` toggle.  The dequant-all path is available as
  ``decode_k_full`` for debugging only and is not called during inference.

State
-----
    state = cache.state()        # serialisable dict (numpy arrays)
    cache2 = KVCompressor.from_state(state, config)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Iterator, List, Optional, Tuple

import mlx.core as mx
import numpy as np

from turboquant.config import TurboQuantConfig
from turboquant.core.pipeline import TurboQuantPipeline
from turboquant.runtime.layout import ensure_layout


# ── View type ─────────────────────────────────────────────────────────────────

@dataclass
class TurboQuantKeysView:
    """Reference into a KVCompressor's compressed K buffer.

    Passed to streaming-attention loops in place of a dense K tensor.
    ``cache.iter_rotated_kv_blocks(view)`` yields decoded K/V blocks.
    """
    cache: "KVCompressor"
    start: int
    end: int
    d_head: int
    block_tokens: int


# ── KVCompressor ──────────────────────────────────────────────────────────────

class KVCompressor:
    """Production compressed KV cache.

    Parameters
    ----------
    config:    TurboQuantConfig
    layer_id:  Layer index passed to calibrated quantisers (unused unless
               you use LayerQuantizers from calibration).
    """

    def __init__(
        self,
        config: TurboQuantConfig,
        layer_id: int = 0,
    ) -> None:
        self.config = config
        self.layer_id = layer_id
        self.pipeline = TurboQuantPipeline(config, layer_id)

        # ── Token buffers (None until first update) ──────────────────────────
        # K compressed
        self._k_packed:      Optional[mx.array] = None  # [B,H,cap,nw_k]
        self._k_scales:      Optional[mx.array] = None  # [B,H,cap,ng_k]
        self._resid_vals:    Optional[mx.array] = None  # [B,H,cap,ng_k,topk]
        self._resid_idx:     Optional[mx.array] = None  # [B,H,cap,ng_k,topk]
        # V compressed
        self._v_packed:      Optional[mx.array] = None  # [B,H,cap,nw_v]
        self._v_scales:      Optional[mx.array] = None  # [B,H,cap,ng_v]

        # ── Metadata ─────────────────────────────────────────────────────────
        self.offset:  int = 0     # tokens stored so far
        self._cap:    int = 0     # allocated capacity
        self._dtype:  Optional[str] = None
        self._B:      int = 0
        self._H:      int = 0

    # ── Capacity management ───────────────────────────────────────────────────

    def _ensure_capacity(
        self,
        B: int,
        H: int,
        T: int,
        D: int,
        V: int,
        dtype,
    ) -> None:
        """Allocate or extend compressed buffers if needed."""
        step = self.config.allocation_step
        cfg = self.config

        need = self.offset + T
        if need <= self._cap:
            return  # already enough room

        new_cap = max(self._cap + step, need)

        from turboquant.core.quantizer import _round_up, _codes_per_word
        # Two-phase padding matching quantize_groups:
        #   Phase 1: round up to group boundary
        #   Phase 2: round up to word-packing boundary
        k_d_pad  = _round_up(D, cfg.k_group_size)
        k_cpw    = _codes_per_word(cfg.k_bits)
        k_d_pack = _round_up(k_d_pad, k_cpw)   # must be divisible by k_cpw
        k_nw     = k_d_pack // k_cpw
        k_ng     = k_d_pad // cfg.k_group_size

        v_d_pad  = _round_up(V, cfg.v_group_size)
        v_cpw    = _codes_per_word(cfg.v_bits)
        v_d_pack = _round_up(v_d_pad, v_cpw)
        v_nw     = v_d_pack // v_cpw
        v_ng     = v_d_pad // cfg.v_group_size

        topk    = cfg.residual_topk
        s_dtype = mx.float16 if cfg.scale_dtype == "float16" else mx.bfloat16

        def _extend(old, new_shape):
            if old is None:
                return mx.zeros(new_shape, dtype=mx.uint32)
            # Pad along the token (axis 2) dimension
            extra = new_shape[2] - old.shape[2]
            pad = mx.zeros(
                (old.shape[0], old.shape[1], extra, *old.shape[3:]),
                dtype=old.dtype,
            )
            return mx.concatenate([old, pad], axis=2)

        def _extend_f(old, new_shape):
            if old is None:
                return mx.zeros(new_shape, dtype=s_dtype)
            extra = new_shape[2] - old.shape[2]
            pad = mx.zeros(
                (old.shape[0], old.shape[1], extra, *old.shape[3:]),
                dtype=old.dtype,
            )
            return mx.concatenate([old, pad], axis=2)

        self._k_packed = _extend(self._k_packed, (B, H, new_cap, k_nw))
        self._k_scales = _extend_f(self._k_scales, (B, H, new_cap, k_ng))

        if topk > 0:
            rv_shape = (B, H, new_cap, k_ng, topk)
            ri_shape = (B, H, new_cap, k_ng, topk)
            self._resid_vals = _extend_f(self._resid_vals, rv_shape)
            self._resid_idx  = (
                mx.zeros(ri_shape, dtype=mx.uint8)
                if self._resid_idx is None
                else mx.concatenate(
                    [self._resid_idx,
                     mx.zeros(
                         (B, H, new_cap - self._cap, k_ng, topk),
                         dtype=mx.uint8,
                     )],
                    axis=2,
                )
            )

        if cfg.v_enabled:
            self._v_packed = _extend(self._v_packed, (B, H, new_cap, v_nw))
            self._v_scales = _extend_f(self._v_scales, (B, H, new_cap, v_ng))

        self._cap   = new_cap
        self._B     = B
        self._H     = H
        self._dtype = str(dtype)

    # ── Public API ────────────────────────────────────────────────────────────

    def update_and_fetch(
        self,
        keys:   mx.array,
        values: mx.array,
    ) -> Tuple[TurboQuantKeysView, mx.array]:
        """Compress and store a new block of (K, V) tokens.

        Parameters
        ----------
        keys:   [B, H, T, D]
        values: [B, H, T, D_v]

        Returns
        -------
        (TurboQuantKeysView, values)  — view for streaming attention;
        raw (non-compressed) values of the CURRENT step for the first
        cross-attention step (the rest are decoded on demand).
        """
        keys   = ensure_layout(keys,   "keys")
        values = ensure_layout(values, "values")

        B, H, T, D = keys.shape
        V = values.shape[-1]
        prev = self.offset

        self._ensure_capacity(B, H, T, D, V, keys.dtype)

        # Encode K
        pk, ks, rv, ri = self.pipeline.encode_k(keys)

        # Store K
        self._k_packed[:, :, prev:prev + T, :] = pk
        self._k_scales[:, :, prev:prev + T, :] = ks.astype(self._k_scales.dtype)

        if self.config.residual_topk > 0 and rv is not None:
            self._resid_vals[:, :, prev:prev + T, :, :] = rv
            self._resid_idx[:, :, prev:prev + T, :, :]  = ri

        # Encode V
        if self.config.v_enabled:
            pv, vs = self.pipeline.encode_v(values)
            self._v_packed[:, :, prev:prev + T, :] = pv
            self._v_scales[:, :, prev:prev + T, :] = vs.astype(
                self._v_scales.dtype
            )

        self.offset += T

        view = TurboQuantKeysView(
            cache=self,
            start=0,
            end=self.offset,
            d_head=D,
            block_tokens=self.config.block_tokens,
        )
        return view, values

    def rotate_queries(self, queries: mx.array) -> mx.array:
        """Rotate Q into the same coordinate frame as stored K."""
        return self.pipeline.rotate_queries(queries)

    # Alias used by gemma.py streaming-attention helper.
    def rotate_queries_for_attention(self, queries: mx.array) -> mx.array:
        """Alias for :meth:`rotate_queries` — kept for model-layer compat."""
        return self.rotate_queries(queries)

    def _make_view(self) -> TurboQuantKeysView:
        """Build a view covering the full stored history [0, offset)."""
        return TurboQuantKeysView(
            cache=self,
            start=0,
            end=self.offset,
            d_head=self.pipeline._d_head or 0,
            block_tokens=self.config.block_tokens,
        )

    # ── Streaming decode ──────────────────────────────────────────────────────

    def iter_rotated_kv_blocks(
        self,
        view: TurboQuantKeysView,
        block_tokens: Optional[int] = None,
    ) -> Iterator[Tuple[int, int, mx.array, mx.array]]:
        """Yield (start, end, k_rotated, v) blocks for streaming attention.

        k_rotated is in the rotated coordinate frame.
        Queries must also be rotated via ``rotate_queries`` before attending.
        """
        blk = block_tokens or view.block_tokens or self.config.block_tokens
        cfg = self.config

        for s in range(view.start, view.end, blk):
            e = min(s + blk, view.end)

            # Decode K (stays rotated)
            pk_blk = self._k_packed[:, :, s:e, :]
            ks_blk = self._k_scales[:, :, s:e, :]
            rv_blk = (
                self._resid_vals[:, :, s:e, :, :]
                if cfg.residual_topk > 0
                else None
            )
            ri_blk = (
                self._resid_idx[:, :, s:e, :, :]
                if cfg.residual_topk > 0
                else None
            )
            k_blk = self.pipeline.decode_k_rotated(
                pk_blk, ks_blk, rv_blk, ri_blk
            )

            # Decode V
            if cfg.v_enabled and self._v_packed is not None:
                pv_blk = self._v_packed[:, :, s:e, :]
                vs_blk = self._v_scales[:, :, s:e, :]
                v_blk  = self.pipeline.decode_v(pv_blk, vs_blk)
            else:
                v_blk = mx.zeros_like(k_blk)

            yield s, e, k_blk, v_blk

    # ── Token management ──────────────────────────────────────────────────────

    def trim(self, n: int) -> int:
        """Logically remove the last *n* tokens from the cache.

        This adjusts ``offset`` only — the underlying storage is not
        compacted.  Use this to roll back speculative decodes or to
        implement a sliding-window over stored tokens.

        Parameters
        ----------
        n:
            Number of tokens to trim.  Clamped to ``[0, offset]``.

        Returns
        -------
        int
            The number of tokens actually trimmed (≤ *n*).
        """
        actual = min(max(n, 0), self.offset)
        self.offset -= actual
        return actual

    def iter_blocks(
        self,
        block_tokens: Optional[int] = None,
    ) -> Iterator[Tuple[int, int, mx.array, mx.array]]:
        """Iterate over the full stored history in (K, V) blocks.

        Equivalent to calling ``iter_rotated_kv_blocks`` with a view that
        spans ``[0, offset)``.  Provided as a convenience so callers do
        not need to construct a ``TurboQuantKeysView`` manually.

        Yields
        ------
        (start, end, k_rotated, v)  — same contract as
        ``iter_rotated_kv_blocks``.
        """
        view = self._make_view()
        yield from self.iter_rotated_kv_blocks(view, block_tokens=block_tokens)

    def memory_breakdown(self) -> dict:
        """Return a breakdown of compressed buffer sizes in bytes.

        Returns
        -------
        dict with keys:
            ``k_packed``, ``k_scales``, ``resid_vals``, ``resid_idx``,
            ``v_packed``, ``v_scales``, ``total``  (all in bytes).
            Zero is reported for any buffer that has not been allocated.
        """

        def _nbytes(a) -> int:
            return int(a.nbytes) if a is not None else 0

        breakdown = {
            "k_packed":   _nbytes(self._k_packed),
            "k_scales":   _nbytes(self._k_scales),
            "resid_vals": _nbytes(self._resid_vals),
            "resid_idx":  _nbytes(self._resid_idx),
            "v_packed":   _nbytes(self._v_packed),
            "v_scales":   _nbytes(self._v_scales),
        }
        breakdown["total"] = sum(breakdown.values())
        return breakdown

    # ── Debug helper (not in hot path) ───────────────────────────────────────

    def decode_k_full(self) -> mx.array:
        """Decode the full K history into dense rotated-space tensor.

        [B, H, offset, D_head] — for debugging / perplexity evaluation only.
        """
        pk = self._k_packed[:, :, :self.offset, :]
        ks = self._k_scales[:, :, :self.offset, :]
        rv = (
            self._resid_vals[:, :, :self.offset, :, :]
            if self.config.residual_topk > 0
            else None
        )
        ri = (
            self._resid_idx[:, :, :self.offset, :, :]
            if self.config.residual_topk > 0
            else None
        )
        return self.pipeline.decode_k_rotated(pk, ks, rv, ri)

    # ── State serialisation ───────────────────────────────────────────────────

    def state(self) -> dict:
        """Return a serialisable dict (values are numpy arrays).

        The dict always includes ``schema_version`` so that
        :func:`turboquant.runtime.state.validate_state` can verify
        compatibility before restoration.
        """
        from turboquant.runtime.state import STATE_SCHEMA_VERSION

        def _np(a):
            return np.array(a) if a is not None else None

        T = self.offset

        def _sl(a, n):
            return a[:, :, :n, ...] if a is not None else None

        k_quant = self.pipeline._k_quant
        v_quant = self.pipeline._v_quant
        cfg = self.config

        s = {
            "schema_version": STATE_SCHEMA_VERSION,
            "offset":     self.offset,
            "d_head":     self.pipeline._d_head,
            "d_pad":      self.pipeline._d_pad,
            "v_dim":      self.pipeline._v_dim,
            "v_pad":      self.pipeline._v_pad,
            "k_bits":     cfg.k_bits,
            "k_group_size": cfg.k_group_size,
            "v_bits":     cfg.v_bits,
            "v_group_size": cfg.v_group_size,
            "v_enabled":  cfg.v_enabled,
            "rotation":   cfg.rotation,
            "rotation_seed": cfg.rotation_seed,
            "residual_topk": cfg.residual_topk,
            "scale_dtype": cfg.scale_dtype,
            "v_scale_dtype": cfg.v_scale_dtype,
            "eps":        cfg.eps,
            "k_packed":   _np(_sl(self._k_packed,   T)),
            "k_scales":   _np(_sl(self._k_scales,   T)),
            "resid_vals": _np(_sl(self._resid_vals, T)),
            "resid_idx":  _np(_sl(self._resid_idx,  T)),
            "v_packed":   _np(_sl(self._v_packed,   T)),
            "v_scales":   _np(_sl(self._v_scales,   T)),
            "k_calibrated_scales": _np(k_quant.calibration_state()) if k_quant is not None else None,
            "v_calibrated_scales": _np(v_quant.calibration_state()) if v_quant is not None else None,
        }
        return s

    @classmethod
    def from_state(
        cls,
        state: dict,
        config: TurboQuantConfig,
        layer_id: int = 0,
    ) -> "KVCompressor":
        """Restore a compressor from a state dict produced by ``state()``.

        Calls :func:`turboquant.runtime.state.validate_state` before
        restoring so that version mismatches are caught early with an
        actionable error message.
        """
        from turboquant.runtime.state import validate_state
        validate_state(state, config)

        obj = cls(config, layer_id)
        obj.offset = state["offset"]

        def _mx(a):
            return mx.array(a) if a is not None else None

        obj._k_packed   = _mx(state.get("k_packed"))
        obj._k_scales   = _mx(state.get("k_scales"))
        obj._resid_vals = _mx(state.get("resid_vals"))
        obj._resid_idx  = _mx(state.get("resid_idx"))
        obj._v_packed   = _mx(state.get("v_packed"))
        obj._v_scales   = _mx(state.get("v_scales"))

        # Restore pipeline dimension metadata
        obj.pipeline._d_head = state.get("d_head")
        obj.pipeline._d_pad  = state.get("d_pad")
        obj.pipeline._v_dim  = state.get("v_dim")
        obj.pipeline._v_pad  = state.get("v_pad")

        k_cal = state.get("k_calibrated_scales")
        v_cal = state.get("v_calibrated_scales")
        if k_cal is not None:
            obj.pipeline._get_k_quant().load_calibration_state(k_cal)
        if v_cal is not None:
            obj.pipeline._get_v_quant().load_calibration_state(v_cal)

        if obj._k_packed is not None:
            obj._cap = obj._k_packed.shape[2]
            obj._B   = obj._k_packed.shape[0]
            obj._H   = obj._k_packed.shape[1]

        return obj
