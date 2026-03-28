"""
FixedRotation — deterministic, seed-locked orthogonal rotation.

Design
------
* ``"hadamard"``          Exact Walsh-Hadamard when ``dim`` is a power of two.
                           For other dimensions, build a deterministic
                           Hadamard-seeded orthogonal matrix via QR so the
                           transform remains square and orthogonal.

* ``"random_orthogonal"`` QR decomposition of a seeded Gaussian matrix.
                           Uses NumPy once at construction; the resulting
                           rotation is stored as a frozen MLX array.

* ``"identity"``          No rotation. Useful for ablations / debugging.

All rotation matrices are computed once and never re-randomised.
``save`` / ``load`` persist the raw matrix so calibrated deployments can
reproduce the exact rotation without re-running QR.
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


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _hadamard_pow2_np(n: int) -> np.ndarray:
    """Normalized Hadamard matrix for power-of-two *n*."""
    if not _is_power_of_two(n):
        raise ValueError(f"Hadamard requires power-of-two size, got {n}")
    H = np.array([[1.0]], dtype=np.float32)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return (H / math.sqrt(float(n))).astype(np.float32)


def _fix_qr_signs(Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Make QR output deterministic by forcing positive diagonal signs."""
    diag = np.sign(np.diag(R)).astype(np.float32)
    diag[diag == 0] = 1.0
    return (Q * diag.reshape(1, -1)).astype(np.float32)


def _hadamard_like_orthogonal_np(n: int) -> np.ndarray:
    """Return an orthogonal n×n matrix derived from a Hadamard basis.

    For power-of-two dimensions this is the exact normalized Hadamard matrix.
    For other dimensions we derive a deterministic orthogonal basis from the
    next-power-of-two Hadamard matrix via QR factorization of a structured
    square submatrix. That preserves the key invariant the compressor needs:
    ``R.T @ R == I``.
    """
    if _is_power_of_two(n):
        return _hadamard_pow2_np(n)

    p = _next_pow2(n)
    H = _hadamard_pow2_np(p)
    A = H[:n, :n].astype(np.float32)
    Q, R = np.linalg.qr(A)
    return _fix_qr_signs(Q.astype(np.float32), R.astype(np.float32))


def _random_orthogonal_np(dim: int, seed: int) -> np.ndarray:
    """Return a dim×dim random orthogonal matrix (QR decomposition)."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((dim, dim)).astype(np.float32)
    Q, R = np.linalg.qr(A)
    return _fix_qr_signs(Q.astype(np.float32), R.astype(np.float32))


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

        self._R: Optional[mx.array] = None
        self._RT: Optional[mx.array] = None

        if rotation_type not in ("identity", "hadamard", "random_orthogonal"):
            raise ValueError(
                f"Unknown rotation_type '{rotation_type}'. "
                "Choose 'identity', 'hadamard', or 'random_orthogonal'."
            )

        if rotation_type == "random_orthogonal":
            R_np = _random_orthogonal_np(dim, seed)
            self._R = mx.array(R_np)
            self._RT = mx.array(R_np.T)
        elif rotation_type == "hadamard":
            R_np = _hadamard_like_orthogonal_np(dim)
            self._R = mx.array(R_np)
            self._RT = mx.array(R_np.T)

    def forward(self, x: mx.array) -> mx.array:
        """Rotate x: [..., D] → [..., D]."""
        if self.rotation_type == "identity":
            return x
        return x @ self._R

    def inverse(self, x: mx.array) -> mx.array:
        """Un-rotate x: [..., D] → [..., D]."""
        if self.rotation_type == "identity":
            return x
        return x @ self._RT

    def save(self, path: str) -> None:
        """Save the rotation matrix as a NumPy .npy file."""
        if self.rotation_type == "identity":
            return
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
        obj.rotation_type = "random_orthogonal"
        obj._R = mx.array(R_np)
        obj._RT = mx.array(R_np.T)
        return obj

    def __repr__(self) -> str:
        return (
            f"FixedRotation(dim={self.dim}, type={self.rotation_type!r}, "
            f"seed={self.seed})"
        )
