# TurboQuant

> **Prototype** — working, tested, honest about what isn't done yet.

A compressed KV-cache for Apple Silicon LLMs, built on top of [mlx-lm](https://github.com/ml-explore/mlx-lm).  
Keys are stored at 3-bit with a residual projection sketch. Values at 4-bit.  
Everything runs in pure MLX — no numpy device syncs in the hot path.

```
Dense KV cache (float16, 1K tokens)  →  1024 KB
TurboQuant (3-bit K + 4-bit V)       →   258 KB   (~4× smaller)
```

---

## What it does

| | Dense KVCache | TurboQuantKCache |
|---|---|---|
| K precision | float16 | 3-bit + per-group scale + residual sign sketch |
| V precision | float16 | 4-bit + per-group scale |
| Storage (1K tok, Gemma-2B shape) | 1024 KB | **258 KB** |
| Compression | 1× | **~4×** |
| Decode step (encode only) | baseline 1.32 ms | **0.41 ms (3.3×)** |
| Rotation | — | identity or Hadamard |
| Return modes | dense | `dequant` (dense) or `view` (block streaming) |

Numbers measured on Apple M-series. Your mileage will vary by model and sequence length.

---

## How it works

```
K path per token
────────────────
raw keys  →  [optional rotation]  →  group-wise scalar quant (3-bit)
          →  packed uint32 words
          →  residual = keys - reconstructed
          →  project residual onto random ±1 basis per group
          →  store sign bits (1 bit/group) + scale (8-bit quantized)

V path per token
────────────────
raw values  →  group-wise scalar quant (4-bit)  →  packed uint32 words
```

At decode time the `view` return mode gives attention a `TurboQuantKeysView`.  
Gemma's attention uses this to stream K/V blocks without reconstructing the  
full history — a standard online-softmax loop over compressed blocks.

**Bit-packing** is fully vectorised in MLX (single broadcast `left_shift` +  
`sum` kernel). No `np.asarray()` calls, no GPU→CPU sync.

---

## Status

| Component | State |
|---|---|
| `TurboQuantConfig` | ✅ stable |
| `TurboQuantKCache` | ✅ working, 18/18 tests |
| `TurboQuantKeysView` + block iterator | ✅ working |
| Pure-MLX bit-packing (no numpy sync) | ✅ done |
| Gemma attention integration | ✅ working |
| `generate.py` hook (`maybe_turboquant_k_cache`) | ✅ wired |
| State / `from_state` round-trip | ✅ tested |
| Residual projection estimator | ⚠️ provisional — simple sign sketch only |
| Fused compressed-domain attention kernel | ❌ not yet |
| Hadamard rotation quality | ⚠️ untested at scale |
| Other architectures (Llama, Qwen3, …) | ❌ needs per-arch attention patch |

This is a **working prototype**, not a production drop-in. The compression ratio  
is real. The quality impact at scale is **not yet measured**.

---

## Quick start

```python
from mlx_lm.models.cache import TurboQuantConfig, TurboQuantKCache

cache = TurboQuantKCache(
    TurboQuantConfig(
        main_bits=3,       # K bit-width
        group_size=64,     # quantisation group size
        rotation="identity",   # or "hadamard"
        return_mode="view",    # streaming attention mode
        v_bits=4,
        v_group_size=64,
        v_enabled=True,
    )
)
```

Pass it as the cache argument to any patched model layer.  
In `generate.py`, use `maybe_turboquant_k_cache()` to upgrade an existing  
`KVCache` after a threshold number of tokens:

```python
from mlx_lm.generate import maybe_turboquant_k_cache

# Upgrade after 256 prefill tokens
maybe_turboquant_k_cache(
    model,
    threshold=256,
    turboquant_main_bits=3,
    turboquant_group_size=64,
)
```

---

## Running the tests

```bash
python -m pytest tests/ -v
# 18 passed in ~5s
```

```bash
python _bench.py
# === TurboQuantKCache decode-step latency ===
#   dequant mode (encode only)      0.55 ms/step  (16+N tokens, 100 reps)
#   view mode   (encode only)       0.41 ms/step  (16+N tokens, 100 reps)
```

---

## Storage breakdown (1024 tokens, 2 KV heads, head_dim=128)

```
k_codes            104 KB   ← 3-bit packed codes
k_scales             8 KB   ← per-group float16 scales
k_resid_scale_q      4 KB   ← 8-bit quantised residual scales
k_resid_scale_max    4 KB   ← per-token scale max
k_resid_proj_signs   2 KB   ← 1 bit per group sign
v_codes            128 KB   ← 4-bit packed codes
v_scales             8 KB   ← per-group float16 scales
─────────────────────────
total              258 KB   (vs 1024 KB dense)
```

---

## What's still weak

- **Residual projection** — the current sketch is a single random ±1 basis  
  vector per group. It recovers a fraction of the residual energy but isn't  
  optimal. A proper learned or structured projector would help.
- **No quality benchmarks** — perplexity / generation quality vs dense cache  
  has not been measured. Don't assume 3-bit is good enough for your use case  
  without testing.
- **Single architecture** — only Gemma's attention has the streaming softmax  
  branch. Every other architecture still needs its own patch.
- **No fused kernel** — attention still iterates over compressed blocks in  
  Python. A Metal kernel for compressed-domain dot products would be the next  
  real speedup.

---

## Requirements

- macOS (Apple Silicon)
- Python ≥ 3.9
- `mlx >= 0.29.3`
- `mlx-lm >= 0.29.1`

---

## Repo layout

```
mlx_lm/
  models/
    cache.py      ← TurboQuantConfig, TurboQuantKeysView, TurboQuantKCache
    gemma.py      ← _expand_kv_heads, _streaming_softmax_attention patch
  generate.py     ← maybe_turboquant_k_cache hook
tests/
  test_turboquant_e2e.py       ← 4 end-to-end generate tests
  test_turboquant_gemma.py     ← 8 cache + attention tests
  test_turboquant_generate.py  ← 6 upgrade-hook tests
_bench.py         ← decode-step latency benchmark
Quant             ← original design notes
```