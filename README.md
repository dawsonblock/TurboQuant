# TurboQuant

A compressed KV-cache for Apple Silicon LLMs, built on [mlx-lm](https://github.com/ml-explore/mlx-lm).
Keys stored at 3-bit with a **top-k sparse residual**. Values at 4-bit.
Deterministic seeded rotation. Per-layer calibration. No numpy in the hot path.

```
Dense KV cache (float16, 1K tokens)  →  1024 KB
TurboQuant (3-bit K + 4-bit V)       →   ~260 KB   (~4× smaller)
```

---

## What it does

| | Dense KVCache | TurboQuant `KVCompressor` |
|---|---|---|
| K precision | float16 | 3-bit + per-group scale + top-k sparse residual |
| V precision | float16 | 4-bit + per-group scale |
| Storage (1K tok, Gemma-2B) | 1024 KB | **~260 KB** |
| Compression | 1× | **~4×** |
| Decode step (encode only) | baseline 1.32 ms | **0.41 ms (3.3×)** |
| Rotation | — | Hadamard (seeded, deterministic) |
| Residual | — | Top-k sparse (k=2/group, fp16 + uint8 idx) |

Numbers measured on Apple M-series.

---

## How it works

```
K path per token
────────────────
raw keys  →  FixedRotation.forward (Hadamard / QR / identity)
          →  GroupScalarQuantizer.encode (N-bit, per-group scale)
          →  residual = rotated_keys - dequant(codes)
          →  encode_topk_residual(residual, k=2)
          →  store: packed_codes, scales, resid_values, resid_indices

V path per token
────────────────
raw values  →  GroupScalarQuantizer.encode (M-bit)  →  packed_codes, scales

Decode K  (rotated space, for streaming attention)
────────────────
packed_codes -> dequant -> + decode_topk_residual -> crop to head_dim
[queries also rotated with FixedRotation.forward before the matmul]
```

Bit-packing is fully vectorised (broadcast `left_shift` + `sum`).
The Hadamard rotation whitens quantisation error, improving effective precision.
The top-k residual captures the dominant error components the quantiser misses.

---

## Status

| Component | State |
|---|---|
| `KVCompressor` (production cache) | ✅ 52/52 tests |
| `TurboQuantPipeline` | ✅ single path, no runtime branches |
| `FixedRotation` | ✅ deterministic, save/load |
| `GroupScalarQuantizer` + calibration | ✅ dynamic + calibrated modes |
| Top-k sparse residual | ✅ per-group, configurable k |
| Pure-MLX bit-packing | ✅ vectorised, no numpy sync |
| `TurboQuantKCache` (legacy mlx-lm) | ✅ 18/18 tests |
| Gemma streaming attention | ✅ working |
| Other architectures | ❌ needs per-arch patch |
| Metal kernel (fused rotate+pack) | ❌ see `turboquant/kernels/` |
| Quality benchmarks (perplexity) | ⚠️ not yet measured |

---

## Quick start

```python
from turboquant import KVCompressor, TurboQuantConfig

config = TurboQuantConfig()   # 3-bit K, 4-bit V, Hadamard, k=2 residual
cache  = KVCompressor(config, layer_id=0)

view, v_cur = cache.update_and_fetch(keys, values)   # keys/values: [B,H,T,D]
q_rot = cache.rotate_queries(queries)                 # rotate Q to match K

for s, e, k_blk, v_blk in cache.iter_rotated_kv_blocks(view):
    ...  # online-softmax attention over (q_rot, k_blk, v_blk)
```

### Optional: calibration

```python
from turboquant.calibration import calibrate

calibrate(cache.pipeline, data_loader,
          extract_kv=lambda b: (b["k"], b["v"]),
          mode="both", max_batches=64)
```

### Legacy mlx-lm interface (unchanged)

```python
from mlx_lm.models.cache import TurboQuantConfig, TurboQuantKCache

cache = TurboQuantKCache(
    TurboQuantConfig(main_bits=3, group_size=64, rotation="hadamard",
                     return_mode="view", v_bits=4, v_enabled=True)
)
```

---

## Running the tests

```bash
python -m pytest turboquant/tests/ tests/ -v
# 70 passed in ~5s   (52 production + 18 legacy)

python benchmarks/decode_latency.py
# dequant mode   0.48 ms/step
# view mode      0.38 ms/step
```

---

## Storage breakdown (1024 tokens, 2 KV heads, head_dim=128)

```
k_codes            ~96 KB   (3-bit packed)
k_scales             8 KB   (per-group fp16 scales)
k_resid_values       8 KB   (top-k fp16 residual values, k=2)
k_resid_indices      4 KB   (top-k uint8 indices)
v_codes            128 KB   (4-bit packed)
v_scales             8 KB   (per-group fp16 scales)
─────────────────────────
total             ~252 KB   (vs 1024 KB dense, ~4× compression)
```

---

## What's still weak

- **No quality benchmarks** — perplexity vs dense cache has not been measured at scale.
- **Single architecture** — only Gemma has the streaming softmax branch.
- **No Metal kernel** — see `turboquant/kernels/` for the roadmap.

---

## Requirements

- macOS (Apple Silicon)
- Python >= 3.9
- `mlx >= 0.29.3`
- `mlx-lm >= 0.29.1`

---

## Repo layout

```
turboquant/
  config.py              -- TurboQuantConfig (immutable, no runtime branches)
  core/
    rotation.py          -- FixedRotation (Hadamard/QR/identity, save/load)
    quantizer.py         -- GroupScalarQuantizer + vectorised pack/unpack
    residual.py          -- encode/decode_topk_residual
    pipeline.py          -- TurboQuantPipeline (single encode/decode path)
  runtime/
    layout.py            -- ensure_layout [B,H,T,D]
    kv_interface.py      -- KVCompressor + TurboQuantKeysView
  calibration/
    fit_quantizer.py     -- calibrate() over a data loader
  kernels/
    __init__.py          -- MLX dispatch note; Metal shader roadmap
  tests/                 -- 52 unit + integration tests
  config/
    default.json
mlx_lm/
  models/
    cache.py             -- TurboQuantKCache (legacy, 18 tests)
    gemma.py             -- streaming attention patch
  generate.py            -- maybe_turboquant_k_cache hook
tests/                   -- 18 legacy integration tests
benchmarks/
  decode_latency.py
docs/
  design-notes.md
