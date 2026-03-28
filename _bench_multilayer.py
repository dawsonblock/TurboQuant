"""
Multi-layer decode bench — simulates 28 attention layers each doing
one decode step at 26k context, measuring realistic per-token latency.

Measures two things:
  1. The raw cost of 28× decode_all_and_attend (baseline)
  2. Whether mx.eval() between layers causes graph fragmentation overhead
  3. Head expansion overhead (GQA: H_kv=8, H_q=32 → 4× repeat)
"""
import time
import mlx.core as mx
from turboquant.config import TurboQuantConfig
from turboquant.runtime.kv_interface import KVCompressor, TurboQuantKeysView
from turboquant.runtime.attention import _expand_kv_heads

cfg = TurboQuantConfig(
    k_bits=3, k_group_size=64,
    v_bits=4, v_group_size=64, v_enabled=True,
    rotation="hadamard",
    residual_topk=2,
)

# Llama-3.2-3B: 24 layers, H_kv=8, H_q=32, D=128
N_LAYERS   = 24
N_TOKENS   = 26620
B, H_kv, H_q, D = 1, 8, 32, 128

print(f"Building {N_LAYERS}-layer cache at {N_TOKENS} tokens…")
caches = [KVCompressor(cfg) for _ in range(N_LAYERS)]
chunk = 512
keys_all = mx.random.normal((B, H_kv, N_TOKENS, D)).astype(mx.float16)
vals_all = mx.random.normal((B, H_kv, N_TOKENS, D)).astype(mx.float16)

# Build all caches
for i in range(0, N_TOKENS, chunk):
    e = min(i + chunk, N_TOKENS)
    k_chunk = keys_all[:, :, i:e, :]
    v_chunk = vals_all[:, :, i:e, :]
    for c in caches:
        c.update_and_fetch(k_chunk, v_chunk)
mx.eval(*[c._k_packed for c in caches])
print("Done building.")

q = mx.random.normal((B, H_q, 1, D)).astype(mx.float16)

views = [
    TurboQuantKeysView(cache=c, start=0, end=c.offset, d_head=D, block_tokens=cfg.block_tokens)
    for c in caches
]

# Warmup
for _ in range(3):
    for c, v in zip(caches, views):
        q_rot = c.rotate_queries(q)
        out = c.decode_all_and_attend(q_rot, v, scale=D**-0.5)
        # Simulate GQA expand (what SDPA gets after expand)
    mx.eval(out)

# ---- Benchmark: current path (decode all + expand inside SDPA) ----
N = 20
t0 = time.perf_counter()
for _ in range(N):
    for c, v in zip(caches, views):
        q_rot = c.rotate_queries(q)
        out = c.decode_all_and_attend(q_rot, v, scale=D**-0.5)
    mx.eval(out)
t_current = (time.perf_counter() - t0) / N * 1000

print(f"\n--- {N_LAYERS}-layer decode, {N_TOKENS} tokens ---")
print(f"  Current path (eval once per step):     {t_current:.1f} ms  => {1000/t_current:.1f} tok/s")

# ---- Now measure: eval once per LAYER (as model loop likely does) ----
t0 = time.perf_counter()
for _ in range(N):
    for c, v in zip(caches, views):
        q_rot = c.rotate_queries(q)
        out = c.decode_all_and_attend(q_rot, v, scale=D**-0.5)
        mx.eval(out)   # eval every layer
t_eval_every = (time.perf_counter() - t0) / N * 1000

print(f"  eval() every layer (mlx-lm style):     {t_eval_every:.1f} ms  => {1000/t_eval_every:.1f} tok/s")

# ---- Measure: explicit head expand cost ----
k_small = mx.random.normal((B, H_kv, N_TOKENS, D)).astype(mx.float16)
v_small = mx.random.normal((B, H_kv, N_TOKENS, D)).astype(mx.float16)
mx.eval(k_small, v_small)

t0 = time.perf_counter()
for _ in range(N):
    for _ in range(N_LAYERS):
        k_exp = _expand_kv_heads(k_small, H_q)
        v_exp = _expand_kv_heads(v_small, H_q)
    mx.eval(k_exp, v_exp)
t_expand = (time.perf_counter() - t0) / N * 1000
k_exp_mb = B * H_q * N_TOKENS * D * 2 / 1e6
print(f"  head expand only ({H_kv}→{H_q} heads):         {t_expand:.1f} ms  ({k_exp_mb:.0f} MB created, {N_LAYERS} layers)")

# ---- Measure: SDPA with and without pre-expand ----
q_rot = caches[0].rotate_queries(q)
pk = caches[0]._k_packed[:, :, :N_TOKENS, :]
ks = caches[0]._k_scales[:, :, :N_TOKENS, :]
rv = caches[0]._resid_vals[:, :, :N_TOKENS, :, :]
ri = caches[0]._resid_idx[:, :, :N_TOKENS, :, :]
k_all = caches[0].pipeline.decode_k_rotated(pk, ks, rv, ri)
pv = caches[0]._v_packed[:, :, :N_TOKENS, :]
vs = caches[0]._v_scales[:, :, :N_TOKENS, :]
v_all = caches[0].pipeline.decode_v(pv, vs)
mx.eval(k_all, v_all, q_rot)

# Unexpanded SDPA (H_kv heads, but q has H_q — this will error or broadcast)
# so we also test with expanded K/V
k_exp = _expand_kv_heads(k_all, H_q)
v_exp = _expand_kv_heads(v_all, H_q)
q_exp = mx.repeat(q_rot, H_q // H_kv, axis=1)  # just to match shape
mx.eval(k_exp, v_exp)

# Expanded SDPA
t0 = time.perf_counter()
for _ in range(N):
    out = mx.fast.scaled_dot_product_attention(q_exp, k_exp, v_exp, scale=D**-0.5, mask="causal")
    mx.eval(out)
t_sdpa_exp = (time.perf_counter() - t0) / N * 1000

# But wait — decode_all_and_attend passes q[H_q] with k[H_kv], does MLX broadcast?
# Test that:
try:
    out2 = mx.fast.scaled_dot_product_attention(q_rot, k_all, v_all, scale=D**-0.5, mask="causal")
    mx.eval(out2)
    t0 = time.perf_counter()
    for _ in range(N):
        out2 = mx.fast.scaled_dot_product_attention(q_rot, k_all, v_all, scale=D**-0.5, mask="causal")
        mx.eval(out2)
    t_sdpa_gqa = (time.perf_counter() - t0) / N * 1000
    print(f"  SDPA native GQA (q={H_q},kv={H_kv}):       {t_sdpa_gqa:.1f} ms  (NO expand needed!)")
except Exception as ex:
    print(f"  SDPA native GQA: NOT supported ({ex})")
    t_sdpa_gqa = None

print(f"  SDPA with pre-expanded K/V ({H_q} heads):  {t_sdpa_exp:.1f} ms")
