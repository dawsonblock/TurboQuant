"""
Streaming attention decode throughput benchmark.

Measures tokens-per-second for the turboquant_streaming_attention path
versus a naive full-materialise baseline, across sequence lengths.

Usage:
    python benchmarks/bench_decode_streaming.py

Sample output:
  === Streaming attention decode latency ===
  seq_len  block_tokens  ms_streaming  ms_baseline  speedup
  --------------------------------------------------------
      256           64         0.45         0.38      0.84x
      ...
"""
import sys
import os
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import mlx.core as mx
from mlx_lm.models.cache import TurboQuantKCache, TurboQuantConfig
from turboquant.runtime.attention import turboquant_streaming_attention

# ---------------------------------------------------------------------------
# Dims
# ---------------------------------------------------------------------------

BATCH = 1
N_HEADS = 8
N_KV_HEADS = 8
HEAD_DIM = 64
SEQ_LENS = [128, 256, 512, 1024]
BLOCK_TOKENS_LIST = [64, 128]
REPS = 30


def _make_q():
    q = mx.random.normal([BATCH, N_HEADS, 1, HEAD_DIM], dtype=mx.float16)
    mx.eval(q)
    return q


def _make_kv(T: int):
    k = mx.random.normal([BATCH, N_KV_HEADS, T, HEAD_DIM], dtype=mx.float16)
    v = mx.random.normal([BATCH, N_KV_HEADS, T, HEAD_DIM], dtype=mx.float16)
    mx.eval(k, v)
    return k, v


def _fill_cache(T: int, block_tokens: int):
    """Fill a view-mode TurboQuant cache and return (cache, keys_view)."""
    cfg = TurboQuantConfig(
        main_bits=3,
        group_size=64,
        return_mode="view",
        block_tokens=block_tokens,
    )
    tq = TurboQuantKCache(cfg)
    kp, vp = _make_kv(T)
    view, _ = tq.update_and_fetch(kp, vp)
    mx.eval(tq.k_codes)
    return tq, view


def _baseline_attn(q, k, v) -> mx.array:
    """Simple sdpa with materialised kv (no streaming)."""
    scale = HEAD_DIM ** -0.5
    k_t = k.swapaxes(-1, -2)
    scores = (q @ k_t) * scale
    weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
    return weights @ v


def main():
    print("=== Streaming attention decode latency ===\n")
    hdr = (
        f"{'seq_len':>7}  {'block_tokens':>12}  "
        f"{'ms_streaming':>13}  {'ms_baseline':>12}  {'speedup':>8}"
    )
    print(hdr)
    print("-" * len(hdr))

    scale = HEAD_DIM ** -0.5

    for T in SEQ_LENS:
        for bt in BLOCK_TOKENS_LIST:
            # ---- streaming path ----
            tq, view = _fill_cache(T, bt)
            q = _make_q()

            # warmup
            for _ in range(3):
                out = turboquant_streaming_attention(q, view, scale=scale)
                mx.eval(out)

            t0 = time.perf_counter()
            for _ in range(REPS):
                out = turboquant_streaming_attention(q, view, scale=scale)
                mx.eval(out)
            ms_stream = (time.perf_counter() - t0) / REPS * 1000

            # ---- baseline: materialise K then standard sdpa ----
            kp, vp = _make_kv(T)
            mx.eval(kp, vp)

            for _ in range(3):
                out2 = _baseline_attn(q, kp, vp)
                mx.eval(out2)

            t0 = time.perf_counter()
            for _ in range(REPS):
                out2 = _baseline_attn(q, kp, vp)
                mx.eval(out2)
            ms_base = (time.perf_counter() - t0) / REPS * 1000

            speedup = ms_base / ms_stream if ms_stream > 0 else float("inf")

            print(
                f"{T:>7}  {bt:>12}  "
                f"{ms_stream:>13.3f}  {ms_base:>12.3f}  {speedup:>7.2f}x"
            )

    print("\ndone.")


if __name__ == "__main__":
    main()
