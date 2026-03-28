"""
TurboQuantPipeline — single, branch-free encode/decode path.

Architecture
------------
K path (encode):
    [B, H, T, D]
      → ensure_layout
      → FixedRotation.forward          (always; identity is no-op)
      → pad to d_pad (divisible by k_group_size)
      → GroupScalarQuantizer.encode    → packed_k, k_scales
      → compute residual (x_rot_pad - dequant)
      → encode_topk_residual           → resid_vals, resid_idx  (if k > 0)

K path (decode):
    packed_k, k_scales, resid_vals, resid_idx
      → GroupScalarQuantizer.decode    → x_hat [B, H, T, d_pad]
      → + decode_topk_residual         (if k > 0)
      → crop to d_head
      → *** stays in rotated space ***  (queries are also rotated for matmul)

V path (encode/decode): same but NO rotation, no residual.

No Python branches in the hot path: the rotation type and residual_topk are
resolved to concrete operations at ``__init__`` time.
"""
from __future__ import annotations

from typing import Optional, Tuple

import mlx.core as mx

from turboquant.config import TurboQuantConfig
from turboquant.core.rotation import FixedRotation
from turboquant.core.quantizer import (
    GroupScalarQuantizer,
    dequantize_groups,
)
from turboquant.core.residual import (
    encode_topk_residual,
    decode_topk_residual,
)


def _round_up(n: int, multiple: int) -> int:
    return ((n + multiple - 1) // multiple) * multiple


class TurboQuantPipeline:
    """Stateless encode/decode engine.  One instance per model layer.

    The pipeline holds:
    * a ``FixedRotation`` (shared across heads / calls for the same layer)
    * a ``GroupScalarQuantizer`` for K and (optionally) one for V

    It does NOT hold any token buffer — that is the responsibility of
    ``KVCompressor``.
    """

    def __init__(self, config: TurboQuantConfig, layer_id: int = 0) -> None:
        self.config = config
        self.layer_id = layer_id

        # Quantisers — created lazily on first call (we don't know head_dim yet)
        self._k_quant: Optional[GroupScalarQuantizer] = None
        self._v_quant: Optional[GroupScalarQuantizer] = None

        # Rotation — created lazily (need head_dim)
        self._rotation_cache: dict[int, FixedRotation] = {}

        # Cached dims (set on first encode_k call)
        self._d_head: Optional[int] = None
        self._d_pad: Optional[int] = None
        self._v_dim: Optional[int] = None
        self._v_pad: Optional[int] = None

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _get_rotation(self, dim: int) -> FixedRotation:
        if dim not in self._rotation_cache:
            self._rotation_cache[dim] = FixedRotation(
                dim, self.config.rotation_seed, self.config.rotation
            )
        return self._rotation_cache[dim]

    def _get_k_quant(self) -> GroupScalarQuantizer:
        if self._k_quant is None:
            self._k_quant = GroupScalarQuantizer(
                self.config.k_bits, self.config.k_group_size, self.config.eps
            )
        return self._k_quant

    def _get_v_quant(self) -> GroupScalarQuantizer:
        if self._v_quant is None:
            self._v_quant = GroupScalarQuantizer(
                self.config.v_bits, self.config.v_group_size, self.config.eps
            )
        return self._v_quant

    # ── K encode ─────────────────────────────────────────────────────────────

    def encode_k(
        self, keys: mx.array
    ) -> Tuple[
        mx.array,
        mx.array,
        Optional[mx.array],
        Optional[mx.array],
    ]:
        """Encode keys: [B, H, T, D].

        Returns
        -------
        packed_k:    [B, H, T, n_k_words]  uint32
        k_scales:    [B, H, T, n_k_groups] float
        resid_vals:  [B, H, T, n_k_groups, k] float16  | None
        resid_idx:   [B, H, T, n_k_groups, k] uint8    | None
        """
        B, H, T, D = keys.shape
        cfg = self.config

        if self._d_head is None:
            self._d_head = D
            self._d_pad = _round_up(D, cfg.k_group_size)
        d_head, d_pad = self._d_head, self._d_pad

        # Rotate
        rot = self._get_rotation(D)
        y = rot.forward(keys)                          # [B, H, T, D]

        # Pad
        if d_pad > D:
            z = mx.zeros((*y.shape[:-1], d_pad - D), dtype=y.dtype)
            y_pad = mx.concatenate([y, z], axis=-1)
        else:
            y_pad = y

        # Quantise
        quant = self._get_k_quant()
        packed_k, k_scales = quant.encode(y_pad)       # [..., n_words], [..., ng]

        # Residual (topk)
        resid_vals = resid_idx = None
        if cfg.residual_topk > 0:
            y_hat = dequantize_groups(
                packed_k, k_scales, cfg.k_bits, cfg.k_group_size, d_pad
            )
            residual = y_pad - y_hat
            resid_vals, resid_idx = encode_topk_residual(
                residual, cfg.residual_topk, cfg.k_group_size
            )

        return packed_k, k_scales, resid_vals, resid_idx

    # ── K decode (rotated space) ──────────────────────────────────────────────

    def decode_k_rotated(
        self,
        packed_k: mx.array,
        k_scales: mx.array,
        resid_vals: Optional[mx.array],
        resid_idx: Optional[mx.array],
    ) -> mx.array:
        """Decode packed K → rotated space [..., d_head].

        The output stays in the rotated coordinate frame so that attention
        can be computed with correspondingly-rotated queries.
        """
        cfg = self.config
        d_head = self._d_head or 0
        d_pad = self._d_pad or 0

        y_hat = dequantize_groups(
            packed_k, k_scales, cfg.k_bits, cfg.k_group_size, d_pad
        )

        if cfg.residual_topk > 0 and resid_vals is not None:
            residual = decode_topk_residual(
                resid_vals, resid_idx, cfg.k_group_size
            )
            y_hat = y_hat + residual[..., :d_pad]

        return y_hat[..., :d_head]

    # ── V encode / decode ─────────────────────────────────────────────────────

    def encode_v(
        self, values: mx.array
    ) -> Tuple[mx.array, mx.array]:
        """Encode values: [B, H, T, D] → (packed_v, v_scales).

        No rotation, no residual.
        """
        cfg = self.config
        B, H, T, D = values.shape

        if self._v_dim is None:
            self._v_dim = D
            self._v_pad = _round_up(D, cfg.v_group_size)

        d_pad = self._v_pad

        if d_pad > D:
            z = mx.zeros((*values.shape[:-1], d_pad - D), dtype=values.dtype)
            values_pad = mx.concatenate([values, z], axis=-1)
        else:
            values_pad = values

        quant = self._get_v_quant()
        return quant.encode(values_pad)

    def decode_v(
        self,
        packed_v: mx.array,
        v_scales: mx.array,
    ) -> mx.array:
        """Decode packed V → [..., v_dim]."""
        cfg = self.config
        v_dim = self._v_dim or 0
        return dequantize_groups(
            packed_v, v_scales, cfg.v_bits, cfg.v_group_size, v_dim
        )

    # ── Query rotation helper (called by attention side) ─────────────────────

    def rotate_queries(self, queries: mx.array) -> mx.array:
        """Rotate Q into the same space as the stored (encoded) K."""
        D = queries.shape[-1]
        rot = self._get_rotation(D)
        return rot.forward(queries)

    # ── Calibration hooks ─────────────────────────────────────────────────────

    def fit_k(self, data: mx.array) -> None:
        """Calibrate K quantiser from a [N, D] sample."""
        self._get_k_quant().fit(data)

    def fit_v(self, data: mx.array) -> None:
        """Calibrate V quantiser from a [N, D] sample."""
        self._get_v_quant().fit(data)
