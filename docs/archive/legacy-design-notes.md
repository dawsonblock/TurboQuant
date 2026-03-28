> **⚠️ ARCHIVED — legacy design notes**
>
> This document describes the prototype `TurboQuantKCache` implementation that
> lives in `mlx_lm/models/cache.py`.  It uses a different residual encoding
> scheme (group-projection sign sketch) and a different config schema
> (`main_bits`, `group_size`, `return_mode`, …) than the production
> `KVCompressor` in the `turboquant/` package.
>
> For the current architecture see `docs/architecture.md`.

---

Below is a single consolidated TurboQuantKCache scaffold that merges the pieces we built:
	•	packed low-bit K
	•	compact K scales
	•	projected residual sketch for K
	•	packed low-bit V
	•	compact V scales
	•	dense-return mode and view mode
	•	blockwise K/V decode for streaming attention
	•	save/load through state and meta_state

It is still a prototype scaffold, not a verified drop-in. I have not executed it against MLX-LM here.

⸻

mlx_lm/models/cache.py

Add this after the existing cache helpers and before KVCache or wherever you want the new class to live.

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx


@dataclass
class TurboQuantConfig:
    main_bits: int = 3
    group_size: int = 64
    rotation: str = "identity"      # "identity" | "hadamard"
    residual: str = "group_proj"
    return_mode: str = "dequant"    # "dequant" | "view"
    block_tokens: int = 256

    scale_dtype: str = "float16"    # for k_scales
    resid_scale_bits: int = 8

    v_bits: int = 4
    v_group_size: int = 64
    v_scale_dtype: str = "float16"
    v_enabled: bool = True

    eps: float = 1e-6


@dataclass
class TurboQuantKeysView:
    cache: "TurboQuantKCache"
    start: int
    end: int
    d_head: int
    block_tokens: int


class TurboQuantKCache(_BaseCache):
    """
    Prototype TurboQuant-style cache:
      - K: rotated, low-bit packed, projected residual sketch
      - V: low-bit packed, no residual correction
      - returns either dense K/V or a key-view for block streaming attention

    This is a scaffold for MLX-LM integration, not a verified finished implementation.
    """

    step = 256

    def __init__(self, config: Optional[TurboQuantConfig] = None):
        self.config = config or TurboQuantConfig()

        # K storage
        self.k_codes = None                # uint32 [B,H,T,n_words]
        self.k_scales = None               # compact dtype [B,H,T,n_groups]
        self.k_resid_scale_q = None        # uint8 [B,H,T,n_groups]
        self.k_resid_scale_max = None      # compact dtype [B,H,T,1]
        self.k_resid_proj_signs = None     # uint8 packed [B,H,T,ceil(n_groups/8)]

        # V storage
        self.v_codes = None                # uint32 [B,H,T,n_words]
        self.v_scales = None               # compact dtype [B,H,T,v_groups]

        # metadata
        self.offset = 0
        self._d_head = None
        self._d_pad = None
        self._n_groups = None

        self._value_dim = None
        self._v_pad = None
        self._v_groups = None

        self._dtype_name = None
        self._rotation_cache = {}

    # -------------------------------------------------------------------------
    # public cache API
    # -------------------------------------------------------------------------

    def size(self):
        return self.offset

    def empty(self):
        return self.k_codes is None

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    @property
    def nbytes(self):
        total = 0
        for x in (
            self.k_codes,
            self.k_scales,
            self.k_resid_scale_q,
            self.k_resid_scale_max,
            self.k_resid_proj_signs,
            self.v_codes,
            self.v_scales,
        ):
            if x is not None:
                total += x.nbytes
        return total

    def storage_breakdown(self):
        return {
            "k_codes": 0 if self.k_codes is None else self.k_codes.nbytes,
            "k_scales": 0 if self.k_scales is None else self.k_scales.nbytes,
            "k_resid_scale_q": 0 if self.k_resid_scale_q is None else self.k_resid_scale_q.nbytes,
            "k_resid_scale_max": 0 if self.k_resid_scale_max is None else self.k_resid_scale_max.nbytes,
            "k_resid_proj_signs": 0 if self.k_resid_proj_signs is None else self.k_resid_proj_signs.nbytes,
            "v_codes": 0 if self.v_codes is None else self.v_codes.nbytes,
            "v_scales": 0 if self.v_scales is None else self.v_scales.nbytes,
            "total": self.nbytes,
        }

    @property
    def state(self):
        if self.k_codes is None:
            return (None, None, None, None, None, None, None)
        if self.offset == self.k_codes.shape[2]:
            return (
                self.k_codes,
                self.k_scales,
                self.k_resid_scale_q,
                self.k_resid_scale_max,
                self.k_resid_proj_signs,
                self.v_codes,
                self.v_scales,
            )
        return (
            self.k_codes[..., : self.offset, :],
            self.k_scales[..., : self.offset, :],
            self.k_resid_scale_q[..., : self.offset, :],
            self.k_resid_scale_max[..., : self.offset, :],
            self.k_resid_proj_signs[..., : self.offset, :],
            self.v_codes[..., : self.offset, :],
            self.v_scales[..., : self.offset, :],
        )

    @state.setter
    def state(self, v):
        (
            self.k_codes,
            self.k_scales,
            self.k_resid_scale_q,
            self.k_resid_scale_max,
            self.k_resid_proj_signs,
            self.v_codes,
            self.v_scales,
        ) = v
        if self.k_codes is None:
            self.offset = 0
            return
        self.offset = self.k_codes.shape[2]

    @property
    def meta_state(self):
        if self._d_head is None:
            return (
                "0", "", "", "", "", "", "", "", "", "",
                "", "", "", "", "", ""
            )
        return (
            str(self.offset),
            str(self._d_head),
            str(self._d_pad),
            str(self._value_dim),
            str(self._v_pad),
            self._dtype_name or "",
            str(self.config.main_bits),
            str(self.config.group_size),
            self.config.rotation,
            self.config.return_mode,
            self.config.scale_dtype,
            str(self.config.resid_scale_bits),
            str(self.config.v_bits),
            str(self.config.v_group_size),
            self.config.v_scale_dtype,
            "1" if self.config.v_enabled else "0",
        )

    @meta_state.setter
    def meta_state(self, v):
        (
            offset,
            d_head,
            d_pad,
            value_dim,
            v_pad,
            dtype_name,
            main_bits,
            group_size,
            rotation,
            return_mode,
            scale_dtype,
            resid_scale_bits,
            v_bits,
            v_group_size,
            v_scale_dtype,
            v_enabled,
        ) = v

        self.offset = int(offset)
        self._d_head = int(d_head) if d_head else None
        self._d_pad = int(d_pad) if d_pad else None
        self._value_dim = int(value_dim) if value_dim else None
        self._v_pad = int(v_pad) if v_pad else None
        self._dtype_name = dtype_name or None

        self.config = TurboQuantConfig(
            main_bits=int(main_bits),
            group_size=int(group_size),
            rotation=rotation,
            return_mode=return_mode,
            scale_dtype=scale_dtype or "float16",
            resid_scale_bits=int(resid_scale_bits),
            v_bits=int(v_bits),
            v_group_size=int(v_group_size),
            v_scale_dtype=v_scale_dtype or "float16",
            v_enabled=(v_enabled == "1"),
        )

        if self._d_pad is not None:
            self._n_groups = self._d_pad // self.config.group_size
        if self._v_pad is not None:
            self._v_groups = self._v_pad // self.config.v_group_size

    @classmethod
    def from_state(cls, state, meta_state):
        obj = cls()
        obj.state = state
        obj.meta_state = meta_state
        return obj

    def update_and_fetch(self, keys, values):
        """
        keys:   [B,H,T,Dk]
        values: [B,H,T,Dv]
        """
        B, H, T, D = keys.shape
        prev = self.offset

        self._ensure_capacity(
            B=B,
            H=H,
            t_new=T,
            d_head=D,
            v_head_dim=values.shape[-1],
            dtype=keys.dtype,
        )

        y = self._rotate(keys)
        y_pad = self._pad_last_dim(y, self._d_pad)

        packed_codes, scales, resid_scale_q, resid_scale_max, resid_proj_signs = self._encode_k(y_pad)

        self.k_codes[..., prev:prev + T, :] = packed_codes
        self.k_scales[..., prev:prev + T, :] = scales
        self.k_resid_scale_q[..., prev:prev + T, :] = resid_scale_q
        self.k_resid_scale_max[..., prev:prev + T, :] = resid_scale_max
        self.k_resid_proj_signs[..., prev:prev + T, :] = resid_proj_signs

        if self.config.v_enabled:
            v_codes, v_scales = self._encode_v(values)
            self.v_codes[..., prev:prev + T, :] = v_codes
            self.v_scales[..., prev:prev + T, :] = v_scales

        self.offset += T

        v_view = self._decode_v_slice(0, self.offset, values.dtype) if self.config.v_enabled else values
        if self.config.return_mode == "view":
            return self._make_view(), v_view

        k_recon = self._decode_k_slice(0, self.offset)
        return k_recon, v_view

    # -------------------------------------------------------------------------
    # specialization helpers
    # -------------------------------------------------------------------------

    def _make_view(self):
        return TurboQuantKeysView(
            cache=self,
            start=0,
            end=self.offset,
            d_head=self._d_head,
            block_tokens=self.config.block_tokens,
        )

    def rotate_queries_for_attention(self, queries: mx.array) -> mx.array:
        return self._rotate(queries)

    def decode_rotated_k_block(self, start: int, end: int) -> mx.array:
        return self._decode_rotated_k_slice(start, end)

    def iter_rotated_kv_blocks(self, view: TurboQuantKeysView, values_unused=None, block_tokens: Optional[int] = None):
        blk = block_tokens or view.block_tokens or self.config.block_tokens
        for s in range(view.start, view.end, blk):
            e = min(s + blk, view.end)
            k_rot = self.decode_rotated_k_block(s, e)
            v_blk = self._decode_v_slice(s, e, k_rot.dtype) if self.config.v_enabled else None
            yield s, e, k_rot, v_blk

    # -------------------------------------------------------------------------
    # allocation
    # -------------------------------------------------------------------------

    def _ensure_capacity(self, *, B: int, H: int, t_new: int, d_head: int, v_head_dim: int, dtype):
        if self._d_head is None:
            self._d_head = d_head
            self._d_pad = self._round_up(d_head, self.config.group_size)
            self._n_groups = self._d_pad // self.config.group_size
            self._value_dim = v_head_dim
            self._v_pad = self._round_up(v_head_dim, self.config.v_group_size)
            self._v_groups = self._v_pad // self.config.v_group_size
            self._dtype_name = str(dtype)

        prev = self.offset
        need = prev + t_new
        if self.k_codes is not None and need <= self.k_codes.shape[2]:
            return

        n_steps = (self.step + t_new - 1) // self.step
        new_cap = n_steps * self.step

        scale_dtype = self._scale_dtype(dtype)
        v_scale_dtype = self._v_scale_dtype(dtype)

        code_shape = (B, H, new_cap, self._packed_code_words(self._d_pad))
        scale_shape = (B, H, new_cap, self._n_groups)
        resid_scale_shape = (B, H, new_cap, self._n_groups)
        resid_scale_max_shape = (B, H, new_cap, 1)
        resid_proj_sign_shape = (B, H, new_cap, (self._n_groups + 7) // 8)

        v_code_shape = (B, H, new_cap, self._packed_v_words(self._v_pad))
        v_scale_shape = (B, H, new_cap, self._v_groups)

        new_codes = mx.zeros(code_shape, dtype=mx.uint32)
        new_scales = mx.zeros(scale_shape, dtype=scale_dtype)
        new_resid_scale_q = mx.zeros(resid_scale_shape, dtype=mx.uint8)
        new_resid_scale_max = mx.zeros(resid_scale_max_shape, dtype=scale_dtype)
        new_resid_proj_signs = mx.zeros(resid_proj_sign_shape, dtype=mx.uint8)

        new_v_codes = mx.zeros(v_code_shape, dtype=mx.uint32)
        new_v_scales = mx.zeros(v_scale_shape, dtype=v_scale_dtype)

        if self.k_codes is None:
            self.k_codes = new_codes
            self.k_scales = new_scales
            self.k_resid_scale_q = new_resid_scale_q
            self.k_resid_scale_max = new_resid_scale_max
            self.k_resid_proj_signs = new_resid_proj_signs
            self.v_codes = new_v_codes
            self.v_scales = new_v_scales
            return

        if prev % self.step != 0:
            self.k_codes = self.k_codes[..., :prev, :]
            self.k_scales = self.k_scales[..., :prev, :]
            self.k_resid_scale_q = self.k_resid_scale_q[..., :prev, :]
            self.k_resid_scale_max = self.k_resid_scale_max[..., :prev, :]
            self.k_resid_proj_signs = self.k_resid_proj_signs[..., :prev, :]
            self.v_codes = self.v_codes[..., :prev, :]
            self.v_scales = self.v_scales[..., :prev, :]

        self.k_codes = mx.concatenate([self.k_codes, new_codes], axis=2)
        self.k_scales = mx.concatenate([self.k_scales, new_scales], axis=2)
        self.k_resid_scale_q = mx.concatenate([self.k_resid_scale_q, new_resid_scale_q], axis=2)
        self.k_resid_scale_max = mx.concatenate([self.k_resid_scale_max, new_resid_scale_max], axis=2)
        self.k_resid_proj_signs = mx.concatenate([self.k_resid_proj_signs, new_resid_proj_signs], axis=2)
        self.v_codes = mx.concatenate([self.v_codes, new_v_codes], axis=2)
        self.v_scales = mx.concatenate([self.v_scales, new_v_scales], axis=2)

    # -------------------------------------------------------------------------
    # K encode / decode
    # -------------------------------------------------------------------------

    def _encode_k(self, y_pad):
        B, H, T, Dp = y_pad.shape
        G = self.config.group_size
        n_groups = Dp // G

        yg = y_pad.reshape(B, H, T, n_groups, G)

        qmax = max(1, (1 << (self.config.main_bits - 1)) - 1)
        scales_fp = mx.maximum(
            mx.max(mx.abs(yg), axis=-1),
            mx.array(self.config.eps, dtype=y_pad.dtype),
        ) / qmax
        scales_e = scales_fp[..., None]

        q = mx.round(yg / scales_e)
        q = mx.clip(q, -qmax, qmax)
        q_i32 = q.astype(mx.int32)

        codes_dense = (q_i32 + qmax).astype(mx.uint8).reshape(B, H, T, Dp)
        packed_codes = self._pack_main_codes(codes_dense)

        y_hat = (q.astype(y_pad.dtype) * scales_e).reshape(B, H, T, Dp)
        resid = y_pad - y_hat
        resid_g = resid.reshape(B, H, T, n_groups, G)

        basis = self._get_group_basis(G, y_pad.dtype).reshape(1, 1, 1, 1, G)
        proj = mx.sum(resid_g * basis, axis=-1)
        proj_signs = proj >= 0
        proj_scale_fp = mx.abs(proj) / G

        scales = self._cast_scale_out(scales_fp, y_pad.dtype)
        resid_scale_q, resid_scale_max = self._quantize_resid_proj_scale(proj_scale_fp, y_pad.dtype)
        resid_proj_signs = self._pack_group_sign_bits(proj_signs)

        return packed_codes, scales, resid_scale_q, resid_scale_max, resid_proj_signs

    def _decode_rotated_k_slice(self, start: int, end: int):
        packed = self.k_codes[..., start:end, :]
        scales = self._cast_scale_in(self.k_scales[..., start:end, :], mx.float32)
        resid_scale_q = self.k_resid_scale_q[..., start:end, :]
        resid_scale_max = self.k_resid_scale_max[..., start:end, :]
        resid_proj_scale = self._dequantize_resid_proj_scale(resid_scale_q, resid_scale_max, mx.float32)
        resid_proj_signs = self.k_resid_proj_signs[..., start:end, :]

        codes = self._unpack_main_codes(packed, self._d_pad)

        B, H, T, Dp = codes.shape
        G = self.config.group_size
        n_groups = Dp // G
        qmax = max(1, (1 << (self.config.main_bits - 1)) - 1)

        q = codes.astype(mx.int32) - qmax
        q = q.reshape(B, H, T, n_groups, G).astype(mx.float32)
        y_hat = q * scales[..., None]

        proj_sign = self._unpack_group_sign_bits(resid_proj_signs, n_groups).astype(mx.float32)
        proj_sign = proj_sign * 2.0 - 1.0

        basis = self._get_group_basis(G, mx.float32).reshape(1, 1, 1, 1, G)
        resid = proj_sign[..., None] * resid_proj_scale[..., None] * basis

        y = (y_hat + resid).reshape(B, H, T, Dp)
        return y[..., : self._d_head]

    def _decode_k_slice(self, start: int, end: int):
        y = self._decode_rotated_k_slice(start, end)
        return self._inverse_rotate(y)

    # -------------------------------------------------------------------------
    # V encode / decode
    # -------------------------------------------------------------------------

    def _encode_v(self, values: mx.array):
        B, H, T, Dv = values.shape
        G = self.config.v_group_size
        v_pad = self._v_pad
        n_groups = self._v_groups

        if Dv < v_pad:
            values = self._pad_last_dim(values, v_pad)

        vg = values.reshape(B, H, T, n_groups, G)

        qmax = max(1, (1 << (self.config.v_bits - 1)) - 1)
        scales_fp = mx.maximum(
            mx.max(mx.abs(vg), axis=-1),
            mx.array(self.config.eps, dtype=values.dtype),
        ) / qmax
        scales_e = scales_fp[..., None]

        q = mx.round(vg / scales_e)
        q = mx.clip(q, -qmax, qmax)
        q_i32 = q.astype(mx.int32)

        codes_dense = (q_i32 + qmax).astype(mx.uint8).reshape(B, H, T, v_pad)
        packed_codes = self._pack_v_codes(codes_dense)
        scales = scales_fp.astype(self._v_scale_dtype(values.dtype))
        return packed_codes, scales

    def _decode_v_slice(self, start: int, end: int, compute_dtype):
        packed = self.v_codes[..., start:end, :]
        scales = self.v_scales[..., start:end, :]
        if scales.dtype != compute_dtype:
            scales = scales.astype(compute_dtype)

        codes = self._unpack_v_codes(packed, self._v_pad)

        B, H, T, Vp = codes.shape
        G = self.config.v_group_size
        n_groups = self._v_groups
        qmax = max(1, (1 << (self.config.v_bits - 1)) - 1)

        q = codes.astype(mx.int32) - qmax
        q = q.reshape(B, H, T, n_groups, G).astype(compute_dtype)
        v_hat = q * scales[..., None]
        v_hat = v_hat.reshape(B, H, T, Vp)
        return v_hat[..., : self._value_dim]

    # -------------------------------------------------------------------------
    # rotation
    # -------------------------------------------------------------------------

    def _rotate(self, x):
        if self.config.rotation == "identity":
            return x
        R = self._get_rotation(x.shape[-1], x.dtype)
        return mx.matmul(x, R)

    def _inverse_rotate(self, x):
        if self.config.rotation == "identity":
            return x
        R = self._get_rotation(x.shape[-1], x.dtype)
        return mx.matmul(x, R)

    def _get_rotation(self, d_head: int, dtype):
        key = ("rot", d_head, str(dtype), self.config.rotation)
        if key in self._rotation_cache:
            return self._rotation_cache[key]

        if self.config.rotation == "identity":
            R = mx.eye(d_head, dtype=dtype)
        elif self.config.rotation == "hadamard":
            if not self._is_power_of_two(d_head):
                R = mx.eye(d_head, dtype=dtype)
            else:
                R = self._hadamard(d_head, dtype)
        else:
            raise ValueError(f"Unsupported rotation mode: {self.config.rotation}")

        self._rotation_cache[key] = R
        return R

    def _hadamard(self, n: int, dtype):
        H = mx.array([[1.0]], dtype=dtype)
        while H.shape[0] < n:
            top = mx.concatenate([H, H], axis=1)
            bottom = mx.concatenate([H, -H], axis=1)
            H = mx.concatenate([top, bottom], axis=0)
        return H / mx.sqrt(mx.array(float(n), dtype=dtype))

    def _get_group_basis(self, group_size: int, dtype):
        key = ("basis", group_size, str(dtype))
        if key in self._rotation_cache:
            return self._rotation_cache[key]

        rng = np.random.default_rng(0)
        basis = rng.choice([-1.0, 1.0], size=(group_size,), replace=True).astype(np.float32)
        basis = mx.array(basis, dtype=dtype)
        self._rotation_cache[key] = basis
        return basis

    # -------------------------------------------------------------------------
    # quantization helpers
    # -------------------------------------------------------------------------

    def _scale_dtype(self, model_dtype):
        if self.config.scale_dtype == "float16":
            return mx.float16
        if self.config.scale_dtype == "bfloat16":
            return mx.bfloat16
        if self.config.scale_dtype == "model":
            return model_dtype
        raise ValueError(f"Unsupported scale_dtype: {self.config.scale_dtype}")

    def _v_scale_dtype(self, model_dtype):
        if self.config.v_scale_dtype == "float16":
            return mx.float16
        if self.config.v_scale_dtype == "bfloat16":
            return mx.bfloat16
        if self.config.v_scale_dtype == "model":
            return model_dtype
        raise ValueError(f"Unsupported v_scale_dtype: {self.config.v_scale_dtype}")

    def _cast_scale_out(self, x: mx.array, model_dtype):
        return x.astype(self._scale_dtype(model_dtype))

    def _cast_scale_in(self, x: mx.array, compute_dtype):
        return x if x.dtype == compute_dtype else x.astype(compute_dtype)

    def _quantize_resid_proj_scale(self, resid_proj_scale_fp: mx.array, model_dtype):
        levels = (1 << self.config.resid_scale_bits) - 1
        mxs = mx.max(resid_proj_scale_fp, axis=-1, keepdims=True)
        mxs = mx.maximum(mxs, mx.array(self.config.eps, dtype=resid_proj_scale_fp.dtype))
        q = mx.round(resid_proj_scale_fp / mxs * levels)
        q = mx.clip(q, 0, levels).astype(mx.uint8)
        return q, self._cast_scale_out(mxs, model_dtype)

    def _dequantize_resid_proj_scale(self, q: mx.array, mxs: mx.array, compute_dtype):
        levels = float((1 << self.config.resid_scale_bits) - 1)
        qf = q.astype(compute_dtype)
        mxs = self._cast_scale_in(mxs, compute_dtype)
        return qf * (mxs / levels)

    # -------------------------------------------------------------------------
    # bit packing helpers
    # -------------------------------------------------------------------------

    def _codes_per_word(self) -> int:
        return 32 // self.config.main_bits

    def _code_mask(self) -> int:
        return (1 << self.config.main_bits) - 1

    def _round_up_codes(self, x: int) -> int:
        cpw = self._codes_per_word()
        return ((x + cpw - 1) // cpw) * cpw

    def _packed_code_words(self, d_pad: int) -> int:
        return self._round_up_codes(d_pad) // self._codes_per_word()

    def _pack_main_codes(self, codes: mx.array) -> mx.array:
        arr = np.asarray(codes)
        d_pad = arr.shape[-1]
        cpw = self._codes_per_word()
        d_pack = self._round_up_codes(d_pad)
        pad = d_pack - d_pad
        if pad:
            arr = np.pad(arr, [(0, 0)] * (arr.ndim - 1) + [(0, pad)], mode="constant", constant_values=0)
        arr = arr.reshape(*arr.shape[:-1], d_pack // cpw, cpw).astype(np.uint32)

        out = np.zeros(arr.shape[:-1], dtype=np.uint32)
        for i in range(cpw):
            out |= arr[..., i] << np.uint32(i * self.config.main_bits)
        return mx.array(out)

    def _unpack_main_codes(self, packed: mx.array, d_pad: int) -> mx.array:
        arr = np.asarray(packed).astype(np.uint32)
        cpw = self._codes_per_word()
        mask = np.uint32(self._code_mask())

        out = np.empty((*arr.shape, cpw), dtype=np.uint8)
        for i in range(cpw):
            out[..., i] = ((arr >> np.uint32(i * self.config.main_bits)) & mask).astype(np.uint8)

        out = out.reshape(*arr.shape[:-1], arr.shape[-1] * cpw)
        out = out[..., :d_pad]
        return mx.array(out)

    def _v_codes_per_word(self) -> int:
        return 32 // self.config.v_bits

    def _v_code_mask(self) -> int:
        return (1 << self.config.v_bits) - 1

    def _round_up_v_codes(self, x: int) -> int:
        cpw = self._v_codes_per_word()
        return ((x + cpw - 1) // cpw) * cpw

    def _packed_v_words(self, v_pad: int) -> int:
        return self._round_up_v_codes(v_pad) // self._v_codes_per_word()

    def _pack_v_codes(self, codes: mx.array) -> mx.array:
        arr = np.asarray(codes)
        v_pad = arr.shape[-1]
        cpw = self._v_codes_per_word()
        v_pack = self._round_up_v_codes(v_pad)
        pad = v_pack - v_pad
        if pad:
            arr = np.pad(arr, [(0, 0)] * (arr.ndim - 1) + [(0, pad)], mode="constant", constant_values=0)
        arr = arr.reshape(*arr.shape[:-1], v_pack // cpw, cpw).astype(np.uint32)

        out = np.zeros(arr.shape[:-1], dtype=np.uint32)
        for i in range(cpw):
            out |= arr[..., i] << np.uint32(i * self.config.v_bits)
        return mx.array(out)

    def _unpack_v_codes(self, packed: mx.array, v_pad: int) -> mx.array:
        arr = np.asarray(packed).astype(np.uint32)
        cpw = self._v_codes_per_word()
        mask = np.uint32(self._v_code_mask())

        out = np.empty((*arr.shape, cpw), dtype=np.uint8)
        for i in range(cpw):
            out[..., i] = ((arr >> np.uint32(i * self.config.v_bits)) & mask).astype(np.uint8)

        out = out.reshape(*arr.shape[:-1], arr.shape[-1] * cpw)
        out = out[..., :v_pad]
        return mx.array(out)

    def _pack_group_sign_bits(self, signs_bool):
        arr = np.asarray(signs_bool.astype(mx.uint8))
        pad = (-arr.shape[-1]) % 8
        if pad:
            arr = np.pad(arr, [(0, 0)] * (arr.ndim - 1) + [(0, pad)], mode="constant", constant_values=0)
        arr = arr.reshape(*arr.shape[:-1], arr.shape[-1] // 8, 8)
        weights = (1 << np.arange(8, dtype=np.uint8)).reshape((1,) * (arr.ndim - 1) + (8,))
        packed = np.sum(arr * weights, axis=-1, dtype=np.uint16).astype(np.uint8)
        return mx.array(packed)

    def _unpack_group_sign_bits(self, packed, n_groups: int):
        arr = np.asarray(packed)
        bits = ((arr[..., None] >> np.arange(8, dtype=np.uint8)) & 1).astype(np.uint8)
        bits = bits.reshape(*arr.shape[:-1], arr.shape[-1] * 8)
        bits = bits[..., :n_groups]
        return mx.array(bits)

    # -------------------------------------------------------------------------
    # misc helpers
    # -------------------------------------------------------------------------

    def _pad_last_dim(self, x, target):
        pad = target - x.shape[-1]
        if pad <= 0:
            return x
        zeros = mx.zeros((*x.shape[:-1], pad), dtype=x.dtype)
        return mx.concatenate([x, zeros], axis=-1)

    @staticmethod
    def _round_up(x: int, multiple: int) -> int:
        return int(math.ceil(x / multiple) * multiple)

    @staticmethod
    def _is_power_of_two(x: int) -> bool:
        return x > 0 and (x & (x - 1)) == 0


⸻

Add this to KVCache

Inside the existing KVCache class:

def to_turboquant(
    self,
    *,
    main_bits: int = 3,
    group_size: int = 64,
    rotation: str = "identity",
    return_mode: str = "dequant",
    scale_dtype: str = "float16",
    resid_scale_bits: int = 8,
    v_bits: int = 4,
    v_group_size: int = 64,
    v_scale_dtype: str = "float16",
    v_enabled: bool = True,
) -> "TurboQuantKCache":
    tq = TurboQuantKCache(
        TurboQuantConfig(
            main_bits=main_bits,
            group_size=group_size,
            rotation=rotation,
            return_mode=return_mode,
            scale_dtype=scale_dtype,
            resid_scale_bits=resid_scale_bits,
            v_bits=v_bits,
            v_group_size=v_group_size,
            v_scale_dtype=v_scale_dtype,
            v_enabled=v_enabled,
        )
    )
    if self.keys is not None:
        keys = self.keys[..., : self.offset, :]
        values = self.values[..., : self.offset, :]
        tq.update_and_fetch(keys, values)
    return tq


⸻

First run settings

Use these first:

main_bits=3
group_size=64
rotation="identity"
return_mode="view"
resid_scale_bits=8

v_bits=4
v_group_size=64
v_enabled=True

Do not start with Hadamard.

First measurements

Run these before looking at quality:

bd = tq_cache.storage_breakdown()
for k, v in bd.items():
    print(f"{k:20s} {v / (1024**2):8.2f} MB")

Then compare total cache size against dense KVCache.

What is still weak

Three parts are still provisional:
	•	residual projection estimator for K
	•	Python/NumPy bit-packing path
	•	no fused compressed-domain attention kernel

That means this is a working prototype shape, not the final design.

The next useful step is to write the matching Gemma streaming attention helper against this exact class so you can benchmark one full path end to end.Patch gemma.py next.

The current file still has the simple path you targeted earlier: project Q/K/V, apply RoPE, call cache.update_and_fetch(keys, values), then pass the result into scaled_dot_product_attention(...). That makes Gemma the clean first integration point for your consolidated cache class.  ￼

mlx_lm/models/gemma.py

Add this import near the top:

from .cache import TurboQuantKeysView

Add these helpers above class Attention:

def _expand_kv_heads(x: mx.array, target_heads: int) -> mx.array:
    """
    Expand KV heads to query heads for grouped-query attention.
    x: [B, H_kv, T, D]
    """
    h = x.shape[1]
    if h == target_heads:
        return x
    if target_heads % h != 0:
        raise ValueError(f"Cannot expand {h} KV heads to {target_heads} query heads")
    repeats = target_heads // h
    return mx.concatenate([x] * repeats, axis=1)


def _streaming_softmax_attention(
    q_rot: mx.array,
    keys_view: TurboQuantKeysView,
    *,
    scale: float,
) -> mx.array:
    """
    Streaming causal attention over TurboQuant K/V blocks.

    q_rot: [B, H_q, L_q, D]
    returns: [B, H_q, L_q, Dv]
    """
    tq = keys_view.cache
    B, H_q, L_q, _ = q_rot.shape

    q_end = keys_view.end
    q_start = q_end - L_q
    q_pos = mx.arange(q_start, q_end, dtype=mx.int32).reshape(1, 1, L_q, 1)

    m = mx.full((B, H_q, L_q, 1), -1e30, dtype=mx.float32)
    l = mx.zeros((B, H_q, L_q, 1), dtype=mx.float32)
    acc = None

    for s, e, k_rot_blk, v_blk in tq.iter_rotated_kv_blocks(keys_view):
        k_rot_blk = _expand_kv_heads(k_rot_blk, H_q)
        v_blk = _expand_kv_heads(v_blk, H_q)

        qf = q_rot.astype(mx.float32)
        kf = k_rot_blk.astype(mx.float32)
        vf = v_blk.astype(mx.float32)

        scores = mx.matmul(qf, kf.transpose(0, 1, 3, 2)) * scale

        k_pos = mx.arange(s, e, dtype=mx.int32).reshape(1, 1, 1, e - s)
        causal = k_pos <= q_pos
        neg_inf = mx.array(-1e30, dtype=scores.dtype)
        scores = mx.where(causal, scores, neg_inf)

        blk_m = mx.max(scores, axis=-1, keepdims=True)
        new_m = mx.maximum(m, blk_m)

        alpha = mx.exp(m - new_m)
        p = mx.exp(scores - new_m)

        if acc is None:
            Dv = vf.shape[-1]
            acc = mx.zeros((B, H_q, L_q, Dv), dtype=mx.float32)

        l = l * alpha + mx.sum(p, axis=-1, keepdims=True)
        acc = acc * alpha + mx.matmul(p, vf)
        m = new_m

    return acc / mx.maximum(l, mx.array(1e-6, dtype=l.dtype))


def turboquant_streaming_attention(
    queries: mx.array,
    keys_view: TurboQuantKeysView,
    *,
    scale: float,
) -> mx.array:
    tq = keys_view.cache
    q_rot = tq.rotate_queries_for_attention(queries)
    return _streaming_softmax_attention(
        q_rot,
        keys_view,
        scale=scale,
    ).astype(queries.dtype)

Now replace the body of Attention.__call__ with this version:

def __call__(
    self,
    x: mx.array,
    mask: Optional[mx.array] = None,
    cache: Optional[Any] = None,
) -> mx.array:
    B, L, D = x.shape
    queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

    queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
    keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
    values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

    if cache is not None:
        queries = self.rope(queries, offset=cache.offset)
        keys = self.rope(keys, offset=cache.offset)
        keys, values = cache.update_and_fetch(keys, values)
    else:
        queries = self.rope(queries)
        keys = self.rope(keys)

    if isinstance(keys, TurboQuantKeysView):
        output = turboquant_streaming_attention(
            queries,
            keys,
            scale=self.scale,
        )
    else:
        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )

    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return self.o_proj(output)

Why this fits

Gemma’s attention call is still a single local function with no special cache abstractions beyond update_and_fetch(...), so you can branch on TurboQuantKeysView without touching the rest of the model stack. The cache system also still restores classes from state and meta_state, which means your custom cache can stay fully inside the existing prompt-cache mechanism.  ￼

What to change in generate.py

Keep using your cache-upgrade helper. generate.py still constructs prompt caches through make_prompt_cache(...), so the right pattern is still “upgrade an existing KVCache into TurboQuantKCache after a threshold,” not “invent a parallel cache runtime.”  ￼

Use these first-run settings:

turboquant_k_start = 32
main_bits = 3
group_size = 64
rotation = "identity"
return_mode = "view"
resid_scale_bits = 8
v_bits = 4
v_group_size = 64
v_enabled = True

First checks

Run these in order:

bd = tq_cache.storage_breakdown()
for k, v in bd.items():
    print(f"{k:20s} {v / (1024**2):8.2f} MB")

Then compare:
	•	dense cache bytes
	•	TurboQuant cache bytes
	•	one short-generation output
	•	one long prompt with a few decode steps

Likely failure points

The two most likely problems are:
	•	head expansion shape mismatches in grouped-query attention
	•	dtype instability in streaming softmax if you do not keep the accumulator in float32

That is why the helper above forces scores, l, m, and acc through float32.

The next useful patch after this is a small test file for Gemma + TurboQuant view mode, not more architectureAdd a small focused test file first.

That is the next useful patch because Gemma’s attention path is still a single local branch around cache.update_and_fetch(...), and MLX-LM’s cache system still restores custom caches through state, meta_state, and class name. A narrow test file will catch shape regressions, cache round-trip issues, and the TurboQuantKeysView branch before you spend more time on benchmarks.  ￼

tests/test_turboquant_gemma.py

Use this as the first scaffold:

import mlx.core as mx

from mlx_lm.models.cache import KVCache, TurboQuantConfig, TurboQuantKCache, TurboQuantKeysView
from mlx_lm.models.gemma import ModelArgs, Attention


def _make_attention():
    args = ModelArgs(
        model_type="gemma",
        hidden_size=32,
        num_hidden_layers=1,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rms_norm_eps=1e-6,
        vocab_size=128,
    )
    return Attention(args)


def _make_input(batch=1, length=4, hidden=32):
    return mx.arange(batch * length * hidden, dtype=mx.float32).reshape(batch, length, hidden) / 100.0


def _make_tq_cache(return_mode="view"):
    return TurboQuantKCache(
        TurboQuantConfig(
            main_bits=3,
            group_size=8,
            rotation="identity",
            residual="group_proj",
            return_mode=return_mode,
            block_tokens=2,
            scale_dtype="float16",
            resid_scale_bits=8,
            v_bits=4,
            v_group_size=8,
            v_scale_dtype="float16",
            v_enabled=True,
        )
    )


def test_turboquant_cache_update_returns_view():
    cache = _make_tq_cache(return_mode="view")

    keys = mx.zeros((1, 2, 3, 8), dtype=mx.float32)
    values = mx.zeros((1, 2, 3, 8), dtype=mx.float32)

    k_out, v_out = cache.update_and_fetch(keys, values)

    assert isinstance(k_out, TurboQuantKeysView)
    assert k_out.start == 0
    assert k_out.end == 3
    assert v_out.shape == (1, 2, 3, 8)


def test_turboquant_cache_update_returns_dense_when_requested():
    cache = _make_tq_cache(return_mode="dequant")

    keys = mx.zeros((1, 2, 3, 8), dtype=mx.float32)
    values = mx.zeros((1, 2, 3, 8), dtype=mx.float32)

    k_out, v_out = cache.update_and_fetch(keys, values)

    assert not isinstance(k_out, TurboQuantKeysView)
    assert k_out.shape == (1, 2, 3, 8)
    assert v_out.shape == (1, 2, 3, 8)


def test_turboquant_cache_state_roundtrip():
    cache = _make_tq_cache(return_mode="view")

    keys = mx.arange(1 * 2 * 5 * 8, dtype=mx.float32).reshape(1, 2, 5, 8) / 50.0
    values = mx.arange(1 * 2 * 5 * 8, dtype=mx.float32).reshape(1, 2, 5, 8) / 40.0
    cache.update_and_fetch(keys, values)

    restored = TurboQuantKCache.from_state(cache.state, cache.meta_state)

    assert restored.offset == cache.offset
    assert restored._d_head == cache._d_head
    assert restored._value_dim == cache._value_dim
    assert restored.nbytes == cache.nbytes


def test_turboquant_iter_rotated_kv_blocks_covers_full_range():
    cache = _make_tq_cache(return_mode="view")

    keys = mx.arange(1 * 2 * 5 * 8, dtype=mx.float32).reshape(1, 2, 5, 8) / 50.0
    values = mx.arange(1 * 2 * 5 * 8, dtype=mx.float32).reshape(1, 2, 5, 8) / 40.0
    view, _ = cache.update_and_fetch(keys, values)

    spans = []
    for s, e, k_blk, v_blk in cache.iter_rotated_kv_blocks(view):
        spans.append((s, e))
        assert k_blk.shape[2] == e - s
        assert v_blk.shape[2] == e - s

    assert spans == [(0, 2), (2, 4), (4, 5)]


def test_gemma_attention_runs_with_dense_kv_cache():
    attn = _make_attention()
    x = _make_input()

    out = attn(x, cache=KVCache())

    assert out.shape == x.shape


def test_gemma_attention_runs_with_turboquant_view_cache():
    attn = _make_attention()
    x = _make_input()
    cache = _make_tq_cache(return_mode="view")

    out = attn(x, cache=cache)

    assert out.shape == x.shape
    assert cache.offset == x.shape[1]


def test_gemma_attention_incremental_two_step_with_turboquant_view_cache():
    attn = _make_attention()
    cache = _make_tq_cache(return_mode="view")

    x1 = _make_input(length=3)
    x2 = _make_input(length=1)

    y1 = attn(x1, cache=cache)
    y2 = attn(x2, cache=cache)

    assert y1.shape == x1.shape
    assert y2.shape == x2.shape
    assert cache.offset == 4


def test_turboquant_storage_breakdown_has_expected_keys():
    cache = _make_tq_cache(return_mode="view")

    keys = mx.zeros((1, 2, 3, 8), dtype=mx.float32)
    values = mx.zeros((1, 2, 3, 8), dtype=mx.float32)
    cache.update_and_fetch(keys, values)

    bd = cache.storage_breakdown()

    expected = {
        "k_codes",
        "k_scales",
        "k_resid_scale_q",
        "k_resid_scale_max",
        "k_resid_proj_signs",
        "v_codes",
        "v_scales",
        "total",
    }
    assert set(bd.keys()) == expected
    assert bd["total"] >= 0

Why this set first

This file tests the actual fault lines you just introduced:
	•	update_and_fetch() returning either dense keys or TurboQuantKeysView
	•	cache round-trip through from_state(state, meta_state)
	•	block iteration for the streaming path
	•	Gemma attention running with both the stock KVCache and your specialized cache branch  ￼

Two small follow-ups

Add one more test after this if the basics pass:

def test_turboquant_view_cache_trim_reduces_offset_only():
    ...

Then add a very small generate-level smoke test, because generate.py still upgrades caches through helper logic during generation rather than through a separate runtime path.  ￼

What I would not test yet

Do not start with:
	•	output-quality assertions against dense cache
	•	long-context benchmarks
	•	Hadamard rotation

First make sure:
	•	shapes are stable
	•	state restoration works
	•	incremental decode does not crash

The next step after this is a tiny generate.py smoke test that confirms the cache-upgrade hook swaps KVCache to TurboQuantKCache after the threshold..