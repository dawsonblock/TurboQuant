"""
TurboQuantKCache decode-step microbenchmark.

Measures per-token encode latency after a prefill, using the test
fixtures from test_turboquant_gemma.py so the cache config is identical
to the unit tests.
"""
import sys, os, time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tests"))

import mlx.core as mx
from test_turboquant_gemma import _make_tq_cache, _make_kv, PREFILL_LEN

REPS = 100


def bench(label: str, return_mode: str = "dequant") -> None:
    # Warmup
    for _ in range(3):
        tq = _make_tq_cache(return_mode)
        kw, vw = _make_kv(PREFILL_LEN)
        tq.update_and_fetch(kw, vw)
        mx.eval(tq.k_codes)

    # Fresh cache + prefill for timing
    tq = _make_tq_cache(return_mode)
    kp, vp = _make_kv(PREFILL_LEN)
    tq.update_and_fetch(kp, vp)
    mx.eval(tq.k_codes)

    k1, v1 = _make_kv(1)

    t0 = time.perf_counter()
    for _ in range(REPS):
        tq.update_and_fetch(k1, v1)
        mx.eval(tq.k_codes)
    t1 = time.perf_counter()

    ms = (t1 - t0) / REPS * 1000
    print(f"  {label:30s}  {ms:.3f} ms/step  "
          f"({PREFILL_LEN}+N tokens, {REPS} reps)")


print("=== TurboQuantKCache decode-step latency ===")
bench("dequant mode (encode only)", "dequant")
bench("view mode   (encode only)", "view")
print("done")
