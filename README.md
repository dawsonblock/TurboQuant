<div align="center">

# ⚡ TurboQuant

**Research-grade KV-cache compression for Apple Silicon MLX LLMs**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)
[![MLX](https://img.shields.io/badge/MLX-0.29.3%2B-orange)](https://github.com/ml-explore/mlx)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-black)](https://apple.com/mac)

*3-bit keys · 4-bit values · deterministic rotation · top-k sparse residual · no numpy in the hot path*

</div>

---

## What

TurboQuant compresses the KV cache of transformer models running on Apple Silicon via [mlx-lm](https://github.com/ml-explore/mlx-lm). It targets memory reduction first. End-to-end latency depends on model, shape, and decode length, and is not publicly certified by generic CI.

> **⚠️ Current status:** Serious prototype. Gemma and Llama families are wired.
> Compression quality (perplexity impact) has **not** been measured at scale.
> No fused Metal kernel exists yet — the hot path runs in Python-level MLX.
> Do not treat the memory/latency numbers as production benchmarks.
> Supported surface is documented in [docs/supported-surface.md](docs/supported-surface.md). Release gating is documented in [docs/release-checklist.md](docs/release-checklist.md).

```
Dense KV cache (fp16, 1K tokens, 2 heads, head_dim=128)   1024 KB
TurboQuant (3-bit K + 4-bit V, group=64)                   ~252 KB   ▸ ~4× smaller
```

| | Dense | TurboQuant |
|---|:---:|:---:|
| K storage | fp16 | **3-bit** + per-group scale + sparse residual |
| V storage | fp16 | **4-bit** + per-group scale |
| 1K token footprint | 1024 KB | **~252 KB** |
| Encode latency | ~1.3 ms | **~0.4 ms** (3×) |
| Rotation | — | Hadamard / Hadamard-derived orthogonal (deterministic) |
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
- **Hadamard-family whitening** — exact dense Hadamard matrix for power-of-two head dims, or a deterministic Hadamard-derived orthogonal fallback otherwise; the rotation equalises per-dimension variance while preserving `R.T @ R = I`. *Not* a fast butterfly transform — cost is O(d²) per token.
- **Top-k sparse residual** — stores the k=2 largest-magnitude quantisation errors per group (fp16 value + uint8 index); recovers the dominant signal the main quantiser misses
- **Two-phase bit-packing** — pad to group boundary, then to word boundary; handles any bit-width (including 3-bit) for any head-dim
- **Single execution path** — config selects operations once at init; no runtime branches.
- **Versioned state schema** — `state()` dicts carry `schema_version: 2`; `validate_state()` enforces correctness on restore.

---

## Install

```bash
git clone https://github.com/dawsonblock/TurboQuant
cd TurboQuant
python -m pip install -e '.[apple]'
```

`mlx` only installs on Apple Silicon. On non-Apple runners, use the packaging and syntax checks only.

---

## Quick start

### Core interface

```python
from turboquant import KVCompressor, TurboQuantConfig

# Defaults: 3-bit K, 4-bit V, Hadamard-family rotation, k=2 sparse residual
config = TurboQuantConfig()
cache  = KVCompressor(config, layer_id=0)

# Each decode step:
view, v_cur = cache.update_and_fetch(keys, values)  # keys/values: [B, H, T, D]
q_rot       = cache.rotate_queries(queries)          # rotate Q into K's frame

for start, end, k_blk, v_blk in cache.iter_rotated_kv_blocks(view):
    # standard online-softmax attention over (q_rot, k_blk, v_blk)
    ...
```

### Wiring into mlx-lm generation

```python
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.cache_upgrade import upgrade_cache_list
from turboquant.config import TurboQuantConfig

cache = make_prompt_cache(model)
# ... run prefill ...

cfg    = TurboQuantConfig(k_bits=3, k_group_size=64, rotation="hadamard")
events = upgrade_cache_list(cache, k_start=64, config=cfg)
# decode loop continues with TurboQuant cache
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

`turboquant_return_mode` and `turboquant_resid_scale_bits` remain in the legacy shim for backward compatibility, but the production upgrade path ignores them. Real residual behavior is controlled by `residual_topk`.


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
pytest tests/

# Unit tests only  (turboquant package, 52 tests)
pytest tests/unit/

# Integration tests only  (mlx_lm + turboquant, 18 tests)
pytest tests/integration/
```

## Local runtime validation

For real MLX runtime certification on Apple Silicon, run:

```bash
./scripts/validate_apple_silicon.sh
```

Public CI only checks packaging and syntax. It does not certify MLX runtime behavior. See [docs/validation-local.md](docs/validation-local.md).

---

## Benchmarks

```bash
# Memory footprint table (bit-width × sequence length)
python benchmarks/bench_memory_footprint.py

# Encode latency: dense vs TurboQuant
python benchmarks/bench_dense_vs_turboquant.py

# Streaming attention throughput
python benchmarks/bench_decode_streaming.py

# Classic per-step latency
python benchmarks/decode_latency.py
```

Sample output from `bench_memory_footprint.py`:

```
type                      bits  group  tokens   total_MB   bytes/tok   vs_dense
dense (float16)             16     --    1024       2.10        2048       1.0×
TurboQuant k=3b g=64         3     64    1024       0.52         512       4.0×
TurboQuant k=2b g=64         2     64    1024       0.43         416       4.9×
```

---

## Evaluation

```python
from mlx_lm.models.cache import TurboQuantConfig
from turboquant.eval import perplexity_report, drift_report, memory_report

cfg = TurboQuantConfig(main_bits=3, group_size=64)

# Perplexity delta vs dense
ppl = perplexity_report(model, input_ids, turboquant_config=cfg)
# → {'dense_ppl': 12.3, 'tq_ppl': 12.6, 'delta_ppl': 0.3, 'n_tokens': 63}

# Logit-distribution KL divergence
drift = drift_report(model, input_ids, turboquant_config=cfg)
# → {'mean_kl': 0.004, 'max_kl': 0.021, 'n_tokens': 63}

# Cache memory comparison
mem = memory_report(model, input_ids, turboquant_config=cfg)
# → {'dense_cache_bytes': 2097152, 'tq_cache_bytes': 524288, 'ratio': 4.0}
```

See [docs/evaluation.md](docs/evaluation.md) for interpretation guidance.

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
├── config.py                  TurboQuantConfig — production schema
├── core/
│   ├── rotation.py            FixedRotation (Hadamard / QR / identity)
│   ├── quantizer.py           GroupScalarQuantizer + vectorised pack/unpack
│   ├── residual.py            encode_topk_residual / decode_topk_residual
│   └── pipeline.py            TurboQuantPipeline — single encode/decode path
├── runtime/
│   ├── layout.py              ensure_layout [B, H, T, D]
│   ├── kv_interface.py        KVCompressor + TurboQuantKeysView
│   ├── attention.py           turboquant_streaming_attention (shared adapter)
│   └── state.py               STATE_SCHEMA_VERSION + validate_state()
├── eval/
│   ├── perplexity.py          perplexity_from_logits(), perplexity_report()
│   ├── generation_drift.py    logit_kl_divergence(), drift_report()
│   └── memory.py              peak_memory_bytes(), memory_report()
├── calibration/
│   └── fit_quantizer.py       calibrate() over any data iterator
└── kernels/
    └── __init__.py            MLX/Metal dispatch note + shader roadmap

mlx_lm/                        patched mlx-lm
├── models/
│   ├── cache.py               TurboQuantKCache adapter + KVCache helpers
│   ├── gemma.py               wired → turboquant_streaming_attention
│   └── llama.py               wired → turboquant_streaming_attention
├── cache_upgrade.py           upgrade_cache_list() — canonical upgrade API
└── generate.py                maybe_turboquant_k_cache (deprecated compatibility shim)

tests/
├── unit/                      52 turboquant package tests
└── integration/               18 mlx_lm integration tests

benchmarks/
├── decode_latency.py
├── bench_memory_footprint.py
├── bench_dense_vs_turboquant.py
└── bench_decode_streaming.py

docs/
├── architecture.md            component map, data-flow, memory model
├── cache-format.md            state dict schema v1, packed uint32 layout
├── integration.md             step-by-step model wiring guide
└── evaluation.md              metrics, benchmark workflow, thresholds
```

---

## Status

| Component | Status |
|---|:---:|
| `KVCompressor` | ✅ tests 52 / 52 |
| `TurboQuantPipeline` | ✅ single path, no branches |
| `FixedRotation` (Hadamard / QR / identity) | ✅ deterministic, save / load |
| `GroupScalarQuantizer` + offline calibration | ✅ dynamic + calibrated |
| Top-k sparse residual | ✅ per-group, configurable k |
| Pure-MLX bit-packing | ✅ vectorised, no numpy sync |
| Versioned state schema (`schema_version: 2`) | ✅ `validate_state()` enforced |
| `TurboQuantKCache` adapter (legacy API) | ✅ tests 18 / 18 |
| Shared streaming attention adapter | ✅ `turboquant.runtime.attention` |
| Gemma streaming attention | ✅ wired |
| Llama streaming attention | ✅ wired |
| `upgrade_cache_list` cache upgrade API | ✅ canonical, idempotent |
| Eval suite (perplexity / KL drift / memory) | ✅ `turboquant.eval` |
| Benchmarks (memory / latency / streaming) | ✅ `benchmarks/` |
| Architecture + integration docs | ✅ `docs/` |
| Other architectures (Mistral, Phi, …) | ⬜ needs per-arch patch |
| Fused Metal kernel (rotate + pack) | ⬜ see `turboquant/kernels/` |
| Perplexity / quality benchmarks at scale | ⬜ not yet measured |

---

## Limitations

- **Quality unmeasured** — compression ratio is real; perplexity impact at scale has not been benchmarked. Use `turboquant.eval.perplexity_report` and `drift_report` to measure on your data.
- **Gemma + Llama wired** — `turboquant_streaming_attention` is dispatched in both. Adding a new architecture is a [one-function change](docs/integration.md#adding-a-new-model-family).
- **No fused kernel yet** — the block iteration runs in Python-level MLX. A Metal shader fusing rotation + pack would be the next real throughput win.
- **Hadamard is O(d²)** — not a fast butterfly transform. For very large head-dims, `rotation="identity"` is faster with marginally worse compression.

---

## Documentation

| Doc | Contents |
|---|---|
| [docs/architecture.md](docs/architecture.md) | Component map, data-flow diagram, memory model |
| [docs/cache-format.md](docs/cache-format.md) | State dict schema v1, uint32 packing layout |
| [docs/integration.md](docs/integration.md) | Step-by-step wiring guide for new models |
| [docs/evaluation.md](docs/evaluation.md) | Metrics reference, benchmark workflow, thresholds |

---

## Requirements

| | |
|---|---|
| Platform | macOS · Apple Silicon (M1 / M2 / M3 / M4) |
| Python | ≥ 3.9 |
| MLX | ≥ 0.29.3 |
| mlx-lm | ≥ 0.29.1 |
