# TurboQuant Evaluation Guide

> How to measure the quality impact of TurboQuant KV compression.

---

## 1. Quick start

```python
from mlx_lm.models.cache import TurboQuantConfig
from turboquant.eval import perplexity_report, drift_report, memory_report

cfg = TurboQuantConfig(main_bits=3, group_size=64, rotation="identity")

# Perplexity
ppl = perplexity_report(model, input_ids, turboquant_config=cfg)
print(ppl)
# {'dense_ppl': 12.3, 'tq_ppl': 12.6, 'delta_ppl': 0.3, 'n_tokens': 63}

# Logit-distribution drift (KL divergence)
drift = drift_report(model, input_ids, turboquant_config=cfg)
print(drift)
# {'mean_kl': 0.004, 'max_kl': 0.021, 'min_kl': 0.0, 'n_tokens': 63}

# Memory
mem = memory_report(model, input_ids, turboquant_config=cfg)
print(mem)
# {'dense_cache_bytes': 2097152, 'tq_cache_bytes': 524288, 'ratio': 4.0, 'n_layers': 18}
```

---

## 2. Metrics

### 2.1 Perplexity (`turboquant.eval.perplexity`)

**What it measures**: how well the TurboQuant model predicts the next token
compared to a dense-cache baseline.

```
PPL = exp( mean NLL )
delta_ppl = tq_ppl - dense_ppl
```

A `delta_ppl` below **0.5** is generally imperceptible in generation quality.
Values above **2.0** indicate the bit-width is too aggressive for this sequence.

**API**:
```python
perplexity_from_logits(logits, targets) -> float
perplexity_report(model, input_ids, turboquant_config, k_start) -> dict
```

### 2.2 Generation drift (`turboquant.eval.generation_drift`)

**What it measures**: KL divergence between dense and TQ token distributions
at each position.  Unlike perplexity, this does not require ground-truth
targets — it compares the model's beliefs unconditionally.

```
KL(P_dense || P_tq) = sum_v P_dense(v) * (log P_dense(v) - log P_tq(v))
```

A `mean_kl` below **0.01** nats indicates negligible distribution shift.

**API**:
```python
logit_kl_divergence(logits_p, logits_q, temperature) -> mx.array  # [T]
drift_report(model, input_ids, turboquant_config, k_start, temperature) -> dict
```

### 2.3 Memory (`turboquant.eval.memory`)

**What it measures**: total bytes consumed by the KV cache arrays after one
forward pass.

```
ratio = dense_cache_bytes / tq_cache_bytes
```

A ratio of **4×** or higher is achievable with 3-bit K + 4-bit V at
`group_size=64` for sequences longer than 512 tokens.

**API**:
```python
peak_memory_bytes(cache_list) -> int
memory_report(model, input_ids, turboquant_config, k_start) -> dict
```

---

## 3. Benchmarks

The `benchmarks/` directory contains standalone scripts that measure
performance without a full model:

| script | what it measures |
|---|---|
| `bench_memory_footprint.py` | bytes per token across bit-widths and sequence lengths |
| `bench_dense_vs_turboquant.py` | encode latency + memory vs dense KVCache |
| `bench_decode_streaming.py` | streaming attention throughput vs full-materialise |

Run any benchmark with:
```bash
python benchmarks/<script>.py
```

---

## 4. Recommended evaluation workflow

1. **Sanity-check memory** with `bench_memory_footprint.py` — verify ratio
   matches theory for your head_dim and bit-width.
2. **Check generation drift** with `drift_report` on a short held-out
   sequence.  `mean_kl < 0.01` is a good pass criterion.
3. **Measure perplexity delta** on your target corpus.  `delta_ppl < 0.5`
   is generally acceptable.
4. **Profile latency** with `bench_decode_streaming.py` to verify the
   streaming attention path is not slower than the dense baseline.

---

## 5. Interpreting results

| metric | typical good range | action if outside range |
|---|---|---|
| `delta_ppl` | < 0.5 | increase `k_bits` or `group_size` |
| `mean_kl` | < 0.01 | increase `k_bits` or disable `residual_topk=0` → `residual_topk=2` |
| memory ratio | > 3× at T ≥ 512 | expected; if lower, check `v_enabled` |
| encode overhead | < 2× dense | if higher, check `rotation` ("identity" is fastest) |
