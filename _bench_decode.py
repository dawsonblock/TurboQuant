"""
Profile decode path at realistic 26k-token scale.
Shows time breakdown: decode_k, decode_v, sdpa, total per step.
"""
import time
import mlx.core as mx
from turboquant.config import TurboQuantConfig
from turboquant.runtime.kv_interface import KVCompressor, TurboQuantKeysView

# Match test_tq.py config
cfg = TurboQuantConfig(
    k_bits=3, k_group_size=64,
    v_bits=4, v_group_size=64, v_enabled=True,
    rotation="hadamard",
    residual_topk=2,
)

B, H, D = 1, 8, 128   # Llama-3.2-3B: 8 heads, head_dim=128
N_PROMPT = 26620

print(f"Simulating {N_PROMPT}-token KV cache, B={B} H={H} D={D}")
print("Building cache (encoding all prompt tokens)...")

cache = KVCompressor(cfg)

# Encode in chunks to avoid OOM
chunk = 512
keys_all   = mx.random.normal((B, H, N_PROMPT, D)).astype(mx.float16)
values_all = mx.random.normal((B, H, N_PROMPT, D)).astype(mx.float16)

for i in range(0, N_PROMPT, chunk):
    e = min(i + chunk, N_PROMPT)
    view, _ = cache.update_and_fetch(keys_all[:,:,i:e,:], values_all[:,:,i:e,:])
mx.eval(cache._k_packed, cache._k_scales)
print(f"Cache built. offset={cache.offset}")
print(f"Memory: {cache.memory_breakdown()['total'] / 1e6:.1f} MB compressed")

# Dense equivalent for comparison
dense_mb = B * H * N_PROMPT * D * 2 * 2 / 1e6  # fp16 K+V
print(f"Dense equivalent: {dense_mb:.1f} MB  ratio: {dense_mb / (cache.memory_breakdown()['total']/1e6):.1f}x")

# ------ Benchmark single decode step ------
q_single = mx.random.normal((B, H, 1, D)).astype(mx.float16)
view = TurboQuantKeysView(cache=cache, start=0, end=cache.offset, d_head=D, block_tokens=cfg.block_tokens)

# Warmup
for _ in range(3):
    q_rot = cache.rotate_queries(q_single)
    out = cache.decode_all_and_attend(q_rot, view, scale=D**-0.5)
    mx.eval(out)

# Time decode_k alone
t0 = time.perf_counter()
N = 20
for _ in range(N):
    pk = cache._k_packed[:, :, :cache.offset, :]
    ks = cache._k_scales[:, :, :cache.offset, :]
    rv = cache._resid_vals[:, :, :cache.offset, :, :]
    ri = cache._resid_idx[:, :, :cache.offset, :, :]
    k_all = cache.pipeline.decode_k_rotated(pk, ks, rv, ri)
    mx.eval(k_all)
t_decode_k = (time.perf_counter() - t0) / N * 1000

# Time decode_v alone
t0 = time.perf_counter()
for _ in range(N):
    pv = cache._v_packed[:, :, :cache.offset, :]
    vs = cache._v_scales[:, :, :cache.offset, :]
    v_all = cache.pipeline.decode_v(pv, vs)
    mx.eval(v_all)
t_decode_v = (time.perf_counter() - t0) / N * 1000

# Time SDPA alone (with pre-decoded tensors)
q_rot = cache.rotate_queries(q_single)
pv = cache._v_packed[:, :, :cache.offset, :]
vs = cache._v_scales[:, :, :cache.offset, :]
k_all = cache.pipeline.decode_k_rotated(pk, ks, rv, ri)
v_all = cache.pipeline.decode_v(pv, vs)
mx.eval(k_all, v_all, q_rot)

t0 = time.perf_counter()
for _ in range(N):
    out = mx.fast.scaled_dot_product_attention(q_rot, k_all, v_all, scale=D**-0.5, mask="causal")
    mx.eval(out)
t_sdpa = (time.perf_counter() - t0) / N * 1000

# Time full decode_all_and_attend
t0 = time.perf_counter()
for _ in range(N):
    q_rot = cache.rotate_queries(q_single)
    out = cache.decode_all_and_attend(q_rot, view, scale=D**-0.5)
    mx.eval(out)
t_full = (time.perf_counter() - t0) / N * 1000

k_mb = B * H * N_PROMPT * D * 2 / 1e6
print(f"\n--- Timing per decode step ({N_PROMPT} tokens) ---")
print(f"  decode_k:              {t_decode_k:7.1f} ms   ({k_mb:.0f} MB decoded)")
print(f"  decode_v:              {t_decode_v:7.1f} ms")
print(f"  sdpa (metal fused):    {t_sdpa:7.1f} ms")
print(f"  full step total:       {t_full:7.1f} ms")
print(f"  => implied tok/s:      {1000/t_full:7.1f}")

# Cleanup
import os; os.remove("/Users/dawsonblock/Downloads/QUANT-AI/_probe.py")
