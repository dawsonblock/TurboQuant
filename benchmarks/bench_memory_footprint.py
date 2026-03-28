"""
Memory footprint analysis: dense vs TurboQuant across bit widths and seq lens.

Usage:
    python benchmarks/bench_memory_footprint.py

Sample output:
  === KV Cache memory footprint ===
  type             bits  group  tokens  total_MB  bytes/token
  -----------------------------------------------------------
  dense (float16)    16     --     256      4.00      16384
  TurboQuant          4     64     256      0.62       2558
  ...
"""
import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import mlx.core as mx
from mlx_lm.models.cache import TurboQuantKCache, TurboQuantConfig

# ---------------------------------------------------------------------------
# Dims
# ---------------------------------------------------------------------------

BATCH = 1
N_HEADS = 8
HEAD_DIM = 64
TOKEN_COUNTS = [256, 512, 1024, 2048]
BIT_CONFIGS = [
    dict(main_bits=4, group_size=64),
    dict(main_bits=3, group_size=64),
    dict(main_bits=2, group_size=64),
    dict(main_bits=4, group_size=32),
    dict(main_bits=3, group_size=32),
]


def _dense_bytes(T: int) -> int:
    """Bytes for a float16 dense KV cache (K + V)."""
    return BATCH * N_HEADS * T * HEAD_DIM * 2 * 2  # 2 bytes * 2 tensors


def _turboquant_bytes(cfg: TurboQuantConfig, T: int) -> int:
    """Fill a TurboQuantKCache with T tokens, return nbytes."""
    tq = TurboQuantKCache(cfg)
    k = mx.random.normal([BATCH, N_HEADS, T, HEAD_DIM], dtype=mx.float16)
    v = mx.random.normal([BATCH, N_HEADS, T, HEAD_DIM], dtype=mx.float16)
    mx.eval(k, v)
    tq.update_and_fetch(k, v)
    mx.eval(tq.k_codes)
    return tq.nbytes


def _breakdown_row(cfg: TurboQuantConfig, T: int) -> dict:
    tq = TurboQuantKCache(cfg)
    k = mx.random.normal([BATCH, N_HEADS, T, HEAD_DIM], dtype=mx.float16)
    v = mx.random.normal([BATCH, N_HEADS, T, HEAD_DIM], dtype=mx.float16)
    mx.eval(k, v)
    tq.update_and_fetch(k, v)
    mx.eval(tq.k_codes)
    return tq.storage_breakdown()


def main():
    print("=== KV Cache memory footprint ===\n")

    # ---- summary table ----
    hdr = (
        f"{'type':24s}  {'bits':>4}  {'group':>5}  "
        f"{'tokens':>6}  {'total_MB':>9}  {'bytes/tok':>10}  "
        f"{'vs_dense':>9}"
    )
    print(hdr)
    print("-" * len(hdr))

    for T in TOKEN_COUNTS:
        dense_b = _dense_bytes(T)
        print(
            f"{'dense (float16)':24s}  {'16':>4}  {'--':>5}  "
            f"{T:>6}  {dense_b/1e6:>9.2f}  "
            f"{dense_b//T:>10}  {'1.0x':>9}"
        )
        for cfg_kw in BIT_CONFIGS:
            cfg = TurboQuantConfig(return_mode="view", **cfg_kw)
            tq_b = _turboquant_bytes(cfg, T)
            ratio = dense_b / tq_b if tq_b > 0 else float("inf")
            bits = cfg_kw["main_bits"]
            grp  = cfg_kw["group_size"]
            label = f"TurboQuant k={bits}b g={grp}"
            print(
                f"{label:24s}  {bits:>4}  {grp:>5}  "
                f"{T:>6}  {tq_b/1e6:>9.2f}  "
                f"{tq_b//T if T else 0:>10}  "
                f"{ratio:>8.1f}x"
            )
        print()

    # ---- per-buffer breakdown for one representative config ----
    print("\n--- Per-buffer breakdown (3-bit k, group=64, 1024 tokens) ---")
    bd = _breakdown_row(TurboQuantConfig(main_bits=3, group_size=64, return_mode="view"), 1024)
    for name, val in bd.items():
        if name == "total":
            print(f"  {'TOTAL':30s}  {val/1e3:8.1f} kB")
        elif val:
            print(f"  {name:30s}  {val/1e3:8.1f} kB")

    print("\ndone.")


if __name__ == "__main__":
    main()
