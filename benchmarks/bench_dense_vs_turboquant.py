"""
Dense KVCache vs TurboQuantKCache: memory and encode-latency comparison.

Usage:
    python benchmarks/bench_dense_vs_turboquant.py

Prints a table like:

  === Dense vs TurboQuant: memory & encode latency ===
  config                  tokens  dense_MB  tq_MB  ratio  ms_dense  ms_tq
  ----------------------------------------------------------------...
  main_bits=3 group=64      256     4.00     0.42   9.6x    0.12     0.18
  ...
"""
import sys
import os
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import mlx.core as mx
from mlx_lm.models.cache import KVCache, TurboQuantKCache, TurboQuantConfig

# ---------------------------------------------------------------------------
# Config matrix
# ---------------------------------------------------------------------------

CONFIGS = [
    dict(main_bits=4, group_size=64),
    dict(main_bits=3, group_size=64),
    dict(main_bits=2, group_size=64),
    dict(main_bits=3, group_size=32),
]

TOKEN_COUNTS = [256, 512, 1024]

# Model-like dims
N_HEADS = 8
HEAD_DIM = 64
N_KV_HEADS = 8
BATCH = 1
REPS = 20


def _make_kv(T: int):
    k = mx.random.normal([BATCH, N_KV_HEADS, T, HEAD_DIM], dtype=mx.float16)
    v = mx.random.normal([BATCH, N_KV_HEADS, T, HEAD_DIM], dtype=mx.float16)
    mx.eval(k, v)
    return k, v


def _dense_bytes(T: int) -> int:
    # float16: 2 bytes/element
    return BATCH * N_KV_HEADS * T * HEAD_DIM * 2 * 2  # K + V


def _bench_encode(cache_factory, T: int) -> float:
    """Return mean ms per update_and_fetch over REPS."""
    # Warmup
    for _ in range(3):
        c = cache_factory()
        k, v = _make_kv(1)
        c.update_and_fetch(k, v)
        if hasattr(c, "k_codes") and c.k_codes is not None:
            mx.eval(c.k_codes)

    # Prefill
    c = cache_factory()
    kp, vp = _make_kv(T)
    c.update_and_fetch(kp, vp)
    if hasattr(c, "k_codes") and c.k_codes is not None:
        mx.eval(c.k_codes)

    k1, v1 = _make_kv(1)
    t0 = time.perf_counter()
    for _ in range(REPS):
        c.update_and_fetch(k1, v1)
        if hasattr(c, "k_codes") and c.k_codes is not None:
            mx.eval(c.k_codes)
        else:
            mx.eval(c.keys)
    elapsed = (time.perf_counter() - t0) / REPS * 1000
    return elapsed


def main():
    print("=== Dense vs TurboQuant: memory & encode latency ===\n")
    hdr = (
        f"{'config':30s}  {'tokens':>6}  "
        f"{'dense_MB':>8}  {'tq_MB':>7}  {'ratio':>6}  "
        f"{'ms_dense':>9}  {'ms_tq':>7}"
    )
    print(hdr)
    print("-" * len(hdr))

    for cfg_kwargs in CONFIGS:
        cfg_label = "  ".join(f"{k}={v}" for k, v in cfg_kwargs.items())
        tq_cfg = TurboQuantConfig(return_mode="view", **cfg_kwargs)

        for T in TOKEN_COUNTS:
            dense_mb = _dense_bytes(T) / 1e6

            # TurboQuant memory after prefill
            tq = TurboQuantKCache(tq_cfg)
            kp, vp = _make_kv(T)
            tq.update_and_fetch(kp, vp)
            mx.eval(tq.k_codes)
            tq_mb = tq.nbytes / 1e6

            ratio = dense_mb / tq_mb if tq_mb > 0 else float("inf")

            # Latency
            ms_dense = _bench_encode(lambda: KVCache(), T)
            ms_tq = _bench_encode(lambda: TurboQuantKCache(tq_cfg), T)

            print(
                f"{cfg_label:30s}  {T:>6}  "
                f"{dense_mb:>8.2f}  {tq_mb:>7.2f}  {ratio:>6.1f}x  "
                f"{ms_dense:>9.3f}  {ms_tq:>7.3f}"
            )

    print("\ndone.")


if __name__ == "__main__":
    main()
