<div align="center">

# ⚡ TurboQuant

**Production KV-cache compression for Apple Silicon LLMs**

[![Tests](https://img.shields.io/badge/tests-70%20passed-brightgreen)](#running-tests)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)
[![MLX](https://img.shields.io/badge/MLX-0.29.3%2B-orange)](https://github.com/ml-explore/mlx)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-black)](https://apple.com/mac)

*3-bit keys · 4-bit values · deterministic rotation · top-k sparse residual · no numpy in the hot path*

</div>

---

## What

TurboQuant compresses the KV cache of transformer models running on Apple Silicon via [mlx-lm](https://github.com/ml-explore/mlx-lm). It cuts memory ~4× with no perceptible latency cost — the encode step is **3× faster** than a naive baseline.

```
Dense KV cache (fp16, 1K tokens, Gemma-2B)   1024 KB
TurboQuant (3-bit K + 4-bit V)                ~252 KB   ▸ ~4× smaller
```

| | Dense | TurboQuant |
|---|:---:|:---:|
| K storage | fp16 | **3-bit** + per-group scale + sparse residual |
| V storage | fp16 | **4-bit** + per-group scale |
| 1K token footprint | 1024 KB | **~252 KB** |
| Encode latency | 1.32 ms | **0.41 ms** (3.3×) |
| Rotation | — | Hadamard (seeded, deterministic) |
| Residual | — | Top-k sparse (k=2/group) |

> Measured on Apple M-series, bs=1, 2 KV heads, head\_dim=128.

---

## How it works

```
                       K  path
┌──────────┐    ┌───────────────┐    ┌──────────────────────┐    ┌────────┐
│ raw keys │───▶│ FixedRotation   │───▶│ GroupScalarQuantizer │───▶│  packed  │
│ [B,H,T,D]│    │ Hadamard / QR   │    │ N-bit, per-group     │    │  codes   │
└──────────┘    └───────────────┘    └──────────────────────┘    └────────┘
                                               │ residual
                                               ▼
                                    ┌──────────────────────┐
                                    │  encode_topk_residual│
                                    │  top-k values+indices│
                                    └──────────────────────┘

                       V  path
┌────────────┐    ┌──────────────────────┐    ┌────────┐
│ raw values │───▶│ GroupScalarQuantizer │───▶│  packed  │
│ [B,H,T,D]  │    │ M-bit, per-group     │    │  codes   │
└────────────┘    └──────────────────────┘    └────────┘

Decode K (streaming attention)
  packed_codes ──▶ dequant ──▶ + topk_residual ──▶ crop ──▶ [B,H,T,D]
  (queries are rotated with the same FixedRotation before the matmul)
```

**Key design choices:**
- **Hadamard whitening** — orthogonal rotation equalises per-dimension variance, making per-group scalar quantisation nearly optimal
- **Top-k sparse residual** — stores the k=2 largest-magnitude quantisation errors per group (fp16 value + uint8 index); recovers the dominant signal the main quantiser misses
- **Two-phase bit-packing** — pad to group boundary, then to word boundary; handles any bit-width (including 3-bit) for any head-dim
- **Single execution path** — no `return_mode` toggle, no dtype fallbacks; the config selects operations once at init

---

## Install

```bash
git clone https://github.com/dawsonblock/TurboQuant
cd TurboQuant
pip install mlx mlx-lm          # only hard dependencies
```

---

## Quick start

### Production interface

```python
from turboquant import KVCompressor, TurboQuantConfig

# Defaults: 3-bit K, 4-bit V, Hadamard rotation, k=2 sparse residual
config = TurboQuantConfig()
cache  = KVCompressor(config, layer_id=0)

# Each decode step:
view, v_cur = cache.update_and_fetch(keys, values)  # keys/values: [B, H, T, D]
q_rot       = cache.rotate_queries(queries)          # rotate Q into K's frame

for start, end, k_blk, v_blk in cache.iter_rotated_kv_blocks(view):
    # standard online-softmax attention over (q_rot, k_blk, v_blk)
    ...
```

### Optional: offline calibration

```python
from turboquant.calibration import calibrate

calibrate(
    cache.pipeline,
    data_loader,
    extract_kv=lambda batch: (batch["keys"], batch["values"]),
    mode="both",        # "k", "v", or "both"
    max_batches=64,
)
# pipeline now uses fitted per-group scales → lower quantisation error
```

### Tune the config

```python
config = TurboQuantConfig(
    k_bits=4,                          # increase for higher K quality
    residual_topk=4,                   # more residual components → lower error
    rotation="random_orthogonal",      # alternative to Hadamard
    rotation_seed=1337,
    v_enabled=False,                   # disable V compression if headroom exists
)
```

### Legacy mlx-lm cache

```python
from mlx_lm.models.cache import TurboQuantConfig, TurboQuantKCache

cache = TurboQuantKCache(
    TurboQuantConfig(main_bits=3, group_size=64, rotation="hadamard",
                     return_mode="view", v_bits=4, v_enabled=True)
)
```

---

## Running tests

```bash
# Full suite — 70 tests in ~5 s
python -m pytest turboquant/tests/ tests/ -v

# Production package only (52 tests)
python -m pytest turboquant/tests/ -v

# Legacy integration only (18 tests)
python -m pytest tests/ -v
```

```bash
# Latency benchmark
python benchmarks/decode_latency.py
# dequant mode   0.48 ms / step
# view mode      0.38 ms / step
```

---

## Memory breakdown

```
1024 tokens · 2 KV heads · head_dim=128

  k_codes           ~96 KB    3-bit packed uint32
  k_scales            8 KB    per-group fp16 scales
  k_resid_values      8 KB    top-k fp16 residual values  (k=2)
  k_resid_indices     4 KB    top-k uint8 indices
  v_codes           128 KB    4-bit packed uint32
  v_scales            8 KB    per-group fp16 scales
  ──────────────────────────
  total            ~252 KB    vs 1024 KB dense  (4.1× compression)
```

---

## Project layout

```
turboquant/
├── config.py                 TurboQuantConfig — immutable, resolved at init
├── core/
│   ├── rotation.py           FixedRotation (Hadamard · QR · identity)
│   ├── quantizer.py          GroupScalarQuantizer + vectorised pack/unpack
│   ├── residual.py           encode_topk_residual / decode_topk_residual
│   └── pipeline.py           TurboQuantPipeline — single encode/decode path
├── runtime/
│   ├── layout.py             ensure_layout [B, H, T, D]
│   └── kv_interface.py       KVCompressor · TurboQuantKeysView
├── calibration/
│   └── fit_quantizer.py      calibrate() over any data iterator
├── kernels/
│   └── __init__.py           MLX/Metal dispatch note + shader roadmap
├── tests/                    52 unit + integration tests
└── config/default.json

mlx_lm/                       patched mlx-lm (legacy interface)
├── models/cache.py           TurboQuantKCache (18 tests)
├── models/gemma.py           streaming softmax attention
└── generate.py               maybe_turboquant_k_cache hook

tests/                        18 legacy integration tests
benchmarks/decode_latency.py
docs/design-notes.md
```

---

## Status

| Component | Status |
|---|:---:|
| `KVCompressor` | ✅ 52 / 52 tests |
| `TurboQuantPipeline` | ✅ single path, no branches |
| `FixedRotation` (Hadamard · QR · identity) | ✅ deterministic, save / load |
| `GroupScalarQuantizer` + offline calibration | ✅ dynamic + calibrated |
| Top-k sparse residual | ✅ per-group, configurable k |
| Pure-MLX bit-packing | ✅ vectorised, no numpy sync |
| `TurboQuantKCache` (legacy) | ✅ 18 / 18 tests |
| Gemma streaming attention | ✅ working |
| Other architectures | ⬜ needs per-arch patch |
| Fused Metal kernel (rotate + pack) | ⬜ see `turboquant/kernels/` |
| Perplexity / quality benchmarks | ⚠️ not yet measured |

---

## Limitations

- **Quality unmeasured** — compression ratio is real; perplexity impact at scale has not been benchmarked. Don't assume 3-bit is sufficient for your use case without testing.
- **Gemma only** — the streaming softmax attention branch is wired for Gemma. Other architectures need their own patch.
- **No fused kernel yet** — the block iteration runs in Python. A Metal shader fusing rotation + pack would be the next real throughput win.

---

## Requirements

| | |
|---|---|
| Platform | macOS · Apple Silicon (M1 / M2 / M3 / M4) |
| Python | ≥ 3.9 |
| MLX | ≥ 0.29.3 |
| mlx-lm | ≥ 0.29.1 |
