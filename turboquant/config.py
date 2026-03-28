"""
TurboQuant configuration.

All parameters are fixed at construction. The pipeline has no runtime
branches — the config selects the execution path once at init time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class TurboQuantConfig:
    # ── Key quantisation ─────────────────────────────────────────────────────
    k_bits: int = 3             # bits per K-code  (3 or 4 recommended)
    k_group_size: int = 64      # quantisation group width along head_dim

    # ── Value quantisation ───────────────────────────────────────────────────
    v_bits: int = 4
    v_group_size: int = 64
    v_enabled: bool = True

    # ── Rotation ─────────────────────────────────────────────────────────────
    # "hadamard"        — dense Hadamard matrix (built once at init, applied via matmul).
    #                     Orthogonal; equalises per-dimension variance.
    #                     Requires d = 2^k (pads to next power of two if needed).
    #                     Cost is O(d²) per token — not a fast butterfly transform.
    # "random_orthogonal" — QR-decomposed Gaussian, works for any d, slower
    # "identity"        — no rotation; use only for debugging
    rotation: Literal["identity", "hadamard", "random_orthogonal"] = "hadamard"
    rotation_seed: int = 42     # fixed seed → deterministic across runs

    # ── Residual ─────────────────────────────────────────────────────────────
    # Top-k sparse residual stored per group after main quantisation.
    # k=0 → disabled (matches old sign-sketch behaviour minus the sketch).
    # k=2 → 12 B/token/head overhead; recovers most residual energy.
    # k=4 → 24 B/token/head; higher quality, more storage.
    residual_topk: int = 2

    # ── Allocation ───────────────────────────────────────────────────────────
    block_tokens: int = 256     # streaming-attention block size
    allocation_step: int = 512  # token slots added per reallocation

    # ── Numerical ────────────────────────────────────────────────────────────
    eps: float = 1e-6
    scale_dtype: Literal["float16", "bfloat16"] = "float16"
    v_scale_dtype: Literal["float16", "bfloat16"] = "float16"

    def __post_init__(self) -> None:
        if self.k_bits < 2 or self.k_bits > 8:
            raise ValueError(f"k_bits must be in [2, 8], got {self.k_bits}")
        if self.v_bits < 2 or self.v_bits > 8:
            raise ValueError(f"v_bits must be in [2, 8], got {self.v_bits}")
        if self.k_group_size < 1:
            raise ValueError(f"k_group_size must be >= 1, got {self.k_group_size}")
        if self.v_group_size < 1:
            raise ValueError(f"v_group_size must be >= 1, got {self.v_group_size}")
        if self.residual_topk < 0:
            raise ValueError(f"residual_topk must be >= 0, got {self.residual_topk}")
        if self.block_tokens < 1:
            raise ValueError(f"block_tokens must be >= 1, got {self.block_tokens}")
        if self.allocation_step < 1:
            raise ValueError(
                f"allocation_step must be >= 1, got {self.allocation_step}"
            )
        _valid_rotations = ("identity", "hadamard", "random_orthogonal")
        if self.rotation not in _valid_rotations:
            raise ValueError(
                f"rotation must be one of {_valid_rotations}, "
                f"got {self.rotation!r}"
            )
        _valid_dtypes = ("float16", "bfloat16")
        if self.scale_dtype not in _valid_dtypes:
            raise ValueError(
                f"scale_dtype must be one of {_valid_dtypes}, "
                f"got {self.scale_dtype!r}"
            )
        if self.v_scale_dtype not in _valid_dtypes:
            raise ValueError(
                f"v_scale_dtype must be one of {_valid_dtypes}, "
                f"got {self.v_scale_dtype!r}"
            )
