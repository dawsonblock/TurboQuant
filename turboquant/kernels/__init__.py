# turboquant/kernels
#
# Platform: Apple Silicon (MLX / Metal)
# Status:   vectorised MLX ops ARE the kernel layer.
#
# ── What this directory is for ─────────────────────────────────────────────
#
# On CUDA/Triton, custom kernels would live here.  On Apple Silicon we compile
# to Metal via MLX's XLA-style compiler.  The vectorised pack / unpack /
# quantise ops in turboquant/core/quantizer.py are already dispatched as fused
# Metal kernels by the MLX runtime — no hand-written shaders are needed for
# the current performance targets.
#
# ── Current hotspot latency (M-series, bs=1, 2-head Gemma) ─────────────────
#
#   Full dequant slice   0.48 ms / step   (vectorised MLX ops)
#   View (no dequant)    0.38 ms / step   (metadata + slicing)
#
# ── Future Metal shader candidates ─────────────────────────────────────────
#
# 1. Fused rotate+pack — single pass over K avoiding the intermediate buffer.
# 2. Fused unpack+dequant+residual-scatter — avoids two temporary tensors.
# 3. On-device topk scatter — replaces the broadcast-comparison trick in
#    turboquant/core/residual.py with a single Metal scatter_nd call.
#
# ── How to add a custom op ─────────────────────────────────────────────────
#
# MLX exposes `mx.fast.metal_kernel` (experimental, ≥ 0.8) for inlining a
# Metal shader string.  When that API stabilises, fused kernels should be
# wired in here and exposed via __init__.py.
#
# For now this package is intentionally empty except for this README.

__all__: list = []
