# TurboQuant Cache Format

> **Schema version**: 1  
> **Defined in**: `turboquant/runtime/state.py`

---

## 1. State dict (v1)

`KVCompressor.state()` returns a Python dict with the following keys:

| key | type | description |
|---|---|---|
| `schema_version` | `int` | always `1` |
| `offset` | `int` | number of tokens stored |
| `d_head` | `int` | original (un-padded) head dimension |
| `d_pad` | `int` | padded head dimension (multiple of group_size) |
| `v_dim` | `int` | original value dimension |
| `v_pad` | `int` | padded value dimension |
| `k_packed` | `mx.array` uint32 | packed K codes `[B, H, offset, n_words]` |
| `k_scales` | `mx.array` float16 | K scale factors `[B, H, offset, n_groups]` |
| `k_resid_vals` | `mx.array` or `None` | top-k residual values `[B, H, offset, topk]` |
| `k_resid_idx` | `mx.array` or `None` | top-k residual indices `[B, H, offset, topk]` |
| `v_packed` | `mx.array` uint32 or `None` | packed V codes |
| `v_scales` | `mx.array` float16 or `None` | V scale factors |

### Packed format

K codes are packed into uint32 words.  For `k_bits = b` and (padded) head
dimension `d_pad`:

```
codes_per_word = floor(32 / b)
n_words        = ceil(d_pad / codes_per_word)
k_packed.shape = [B, H, T, n_words]
```

Each uint32 stores `codes_per_word` b-bit unsigned integers in the **low bits
first** order:

```
word = code[0] | (code[1] << b) | (code[2] << 2b) | ...
```

Unpacking extracts each code as `(word >> (i * b)) & ((1 << b) - 1)`.

### Scale format

One scale per group:

```
n_groups      = ceil(d_pad / group_size)
k_scales.shape = [B, H, T, n_groups]
```

Dequantisation: `x_float = (code - zero) * scale`  where `zero = (1 << b) / 2`
(symmetric unsigned-to-signed mapping).

---

## 2. Legacy state (7-tuple)

`TurboQuantKCache.state` (property) returns a 7-tuple for backward compatibility
with checkpoints written before schema v1:

```python
(k_codes, k_scales, None, None, None, v_codes, v_scales)
```

Indices 2–4 (residual sign-sketch fields) are always `None` in the production
path.  `TurboQuantKCache.from_state(state, meta_state)` accepts this 7-tuple
plus a 17-element string tuple (`meta_state`) that encodes all config fields as
strings.

### meta_state tuple layout

```
index  field
0      offset (str int)
1      d_head (str int)
2      d_pad  (str int or "")
3      v_dim  (str int or "")
4      v_pad  (str int or "")
5      dtype name (str or "")
6      main_bits
7      group_size
8      rotation
9      return_mode
10     scale_dtype
11     resid_scale_bits
12     v_bits
13     v_group_size
14     v_scale_dtype
15     v_enabled ("1" | "0")
16     block_tokens
```

---

## 3. State validation

`validate_state(state, config=None)` in `turboquant/runtime/state.py` raises
`ValueError` for:

- Missing `schema_version`
- `schema_version` ≠ `STATE_SCHEMA_VERSION` (currently 1)
- Missing required scalar keys
- `k_packed.shape[2]` < `state["offset"]`
- Group count in `k_scales` inconsistent with `config.k_group_size` (if config provided)

---

## 4. Serialisation example

```python
import mlx.core as mx
from turboquant.runtime.kv_interface import KVCompressor
from turboquant.config import TurboQuantConfig

cfg = TurboQuantConfig(k_bits=3, k_group_size=64)
comp = KVCompressor(cfg)
k = mx.random.normal([1, 8, 32, 64], dtype=mx.float16)
v = mx.random.normal([1, 8, 32, 64], dtype=mx.float16)
comp.update_and_fetch(k, v)

# Save
state = comp.state()
mx.savez("kv_state.npz", **{k: v for k, v in state.items()
                              if isinstance(v, mx.array)})

# Restore
restored = KVCompressor.from_state(state, cfg)
assert restored.offset == comp.offset
```
