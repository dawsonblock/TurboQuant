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
    # "hadamard"        — fast Walsh-Hadamard, best quality, requires d = 2^k
    #                     (will pad to next power of two if needed)
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
        if self.residual_topk < 0:
            raise ValueError(f"residual_topk must be >= 0, got {self.residual_topk}")
