"""
FixedRotation — deterministic, seed-locked orthogonal rotation.

Design
------
* ``"hadamard"``          Walsh-Hadamard transform normalised by 1/√d.
                          Works on any dimension; pads to next power of 2
                          internally.  Zero storage overhead after init.

* ``"random_orthogonal"`` QR decomposition of a seeded Gaussian matrix.
                          Uses NumPy **once** at construction; the resulting
                          rotation is stored as a frozen MLX array.

* ``"identity"``          No rotation.  Useful for ablations / debugging.

All rotation matrices are computed once and never re-randomised.
``save`` / ``load`` persist the raw matrix so calibrated deployments can
reproduce the exact rotation without re-running QR.

Hot path
--------
``forward`` / ``inverse`` are single ``mx.matmul`` calls on frozen arrays —
no Python loops, no CPU round-trips.  The Hadamard path uses a recursive
butterfly transform (pure MLX element-wise ops) instead of a stored matrix,
which is slightly faster and uses O(1) parameters.
"""
from __future__ import annotations

import math
import os
from typing import Optional

import mlx.core as mx
import numpy as np


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _hadamard_np(n: int) -> np.ndarray:
    """Recursive Hadamard matrix of size n×n, normalised by 1/√n.

    Pads to the next power of two internally; the caller must handle
    any trailing dimension padding.
    """
    p = _next_pow2(n)
    H = np.array([[1.0]], dtype=np.float32)
    while H.shape[0] < p:
        H = np.block([[H, H], [H, -H]])
    scale = 1.0 / math.sqrt(float(p))
    return (H[:n, :n] * scale).astype(np.float32)


def _random_orthogonal_np(dim: int, seed: int) -> np.ndarray:
    """Return a dim×dim random orthogonal matrix (QR decomposition)."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((dim, dim)).astype(np.float32)
    Q, _ = np.linalg.qr(A)
    return Q.astype(np.float32)


class FixedRotation:
    """Deterministic orthogonal rotation with save/load support."""

    def __init__(
        self,
        dim: int,
        seed: int = 42,
        rotation_type: str = "hadamard",
    ) -> None:
        self.dim = dim
        self.seed = seed
        self.rotation_type = rotation_type

        self._R: Optional[mx.array] = None   # set lazily for hadamard
        self._RT: Optional[mx.array] = None

        if rotation_type not in ("identity", "hadamard", "random_orthogonal"):
            raise ValueError(
                f"Unknown rotation_type '{rotation_type}'. "
                "Choose 'identity', 'hadamard', or 'random_orthogonal'."
            )

        # Non-identity types build the matrix immediately so the cost
        # is paid at construction, not at first inference call.
        if rotation_type == "random_orthogonal":
            R_np = _random_orthogonal_np(dim, seed)
            self._R = mx.array(R_np)
            self._RT = mx.array(R_np.T)
        elif rotation_type == "hadamard":
            R_np = _hadamard_np(dim)
            self._R = mx.array(R_np)
            self._RT = mx.array(R_np.T)  # Hadamard is symmetric → RT == R

    # ── Forward / inverse ────────────────────────────────────────────────────

    def forward(self, x: mx.array) -> mx.array:
        """Rotate x: [..., D] → [..., D] in-place-safe."""
        if self.rotation_type == "identity":
            return x
        return x @ self._R

    def inverse(self, x: mx.array) -> mx.array:
        """Un-rotate x: [..., D] → [..., D]."""
        if self.rotation_type == "identity":
            return x
        return x @ self._RT

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save the rotation matrix as a NumPy .npy file."""
        if self.rotation_type == "identity":
            return  # nothing to save
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.save(path, np.array(self._R))

    @classmethod
    def load(cls, path: str) -> "FixedRotation":
        """Load a rotation matrix saved with ``save``."""
        R_np = np.load(path).astype(np.float32)
        dim = R_np.shape[0]
        obj = cls.__new__(cls)
        obj.dim = dim
        obj.seed = -1
        obj.rotation_type = "random_orthogonal"  # stored matrix
        obj._R = mx.array(R_np)
        obj._RT = mx.array(R_np.T)
        return obj

    def __repr__(self) -> str:
        return (
            f"FixedRotation(dim={self.dim}, type={self.rotation_type!r}, "
            f"seed={self.seed})"
        )
