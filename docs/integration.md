# TurboQuant Integration Guide

> How to wire TurboQuant into an mlx-lm model family.

---

## 1. Concepts

TurboQuant inserts itself into two places:

1. **KV cache** — replace `KVCache` with `TurboQuantKCache` after prefill
2. **Attention** — dispatch to the streaming attention path when the key tensor
   is a `TurboQuantKeysView`

Both hooks are model-agnostic.  The cache upgrade is done once (after prefill);
the attention dispatch is a one-liner inside each model's attention `__call__`.

---

## 2. Cache upgrade

### Recommended (programmatic)

```python
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.cache_upgrade import upgrade_cache_list
from turboquant.config import TurboQuantConfig

cache = make_prompt_cache(model)
# ... run prefill ...
cfg = TurboQuantConfig(k_bits=3, k_group_size=64, rotation="hadamard")
events = upgrade_cache_list(cache, k_start=64, config=cfg)
# decode loop continues with TurboQuant cache
```

`upgrade_cache_list` returns a list of `CacheUpgradeEvent` objects (one per
layer) with `upgraded`, `layer_index`, `old_type`, `new_type`, `offset_at_upgrade`.

### Legacy (deprecated)

`mlx_lm.generate.maybe_turboquant_k_cache` is still importable for backward
compatibility but internally delegates to `upgrade_cache_list`.  New code should
use `upgrade_cache_list` directly.

---

## 3. Attention dispatch

### Step 1 — import the shared adapter

Add this import to your model file:

```python
from turboquant.runtime.attention import maybe_turboquant_attention
```

### Step 2 — dispatch inside attention `__call__`

Replace the attention call with:

```python
def __call__(self, x, mask=None, cache=None):
    q, k, v = ...   # project x

    if cache is not None:
        k, v = cache.update_and_fetch(k, v)

    scale = self.scale   # or head_dim ** -0.5
    attn_out = maybe_turboquant_attention(
        q, k, v, mask, scale,
        fallback=self._dense_attention,
        cache=cache,
    )
    return self.o_proj(attn_out)

def _dense_attention(self, q, k, v, mask, scale):
    # your existing dense attention implementation
    ...
```

`maybe_turboquant_attention` dispatches on `isinstance(k, TurboQuantKeysView)`:

- **TurboQuant path**: calls `turboquant_streaming_attention(q, keys_view, scale=scale)`
- **Dense path**: calls `fallback(q, k, v, mask, scale)`

### Gemma example

`mlx_lm/models/gemma.py` is the reference implementation.  Search for
`maybe_turboquant_attention` to see the exact wiring.

---

## 4. Config mapping (legacy → production)

If you have old code using `TurboQuantConfig(main_bits=3, group_size=64, ...)`:

| legacy field | production field | notes |
|---|---|---|
| `main_bits` | `k_bits` | |
| `group_size` | `k_group_size` | |
| `rotation` | `rotation` | same values |
| `return_mode` | — | adapter-only; production upgrade path always uses streaming view mode |
| `resid_scale_bits` | — | adapter metadata only; production residual behavior is `residual_topk` |
| `residual` | — | ignored |
| `v_bits` | `v_bits` | |
| `v_group_size` | `v_group_size` | |
| `block_tokens` | `block_tokens` | |

`mlx_lm.models.cache.TurboQuantConfig` is a legacy shim that performs this
mapping automatically.

---

## 5. Llama wiring

`mlx_lm/models/llama.py` wires `maybe_turboquant_attention` identically to
Gemma.  See `mlx_lm/models/llama.py` → `Attention.__call__`.

---

## 6. Adding a new model family

1. Add `from turboquant.runtime.attention import maybe_turboquant_attention`
2. In `Attention.__call__`, replace the dense `scaled_dot_product_attention`
   call with `maybe_turboquant_attention(..., fallback=self._dense_attention)`
3. No changes to the cache object are needed — `TurboQuantKCache` is fully
   encapsulated.

---

## 7. Testing

```bash
# Unit tests (turboquant package)
pytest tests/unit/

# Integration tests (mlx_lm + turboquant)
pytest tests/integration/

# Full suite
pytest tests/
```
