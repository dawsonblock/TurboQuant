# Supported surface

This repository does **not** claim broad `mlx_lm` model coverage. The codebase vendors a large upstream tree, but the TurboQuant-specific attention path is only wired and discussed for a narrow slice.

## Supported slice

What this repository currently intends to support:

- Apple Silicon Macs
- Python 3.9+
- MLX runtime installed locally
- Research and local evaluation workflows
- TurboQuant core package: `turboquant/*`
- `mlx_lm` adapter path used to upgrade dense prompt caches into `TurboQuantKCache`
- Llama-family integration path
- Gemma-family integration path

## Not claimed

What is **not** claimed by the current repository state:

- Public CI runtime certification of MLX-backed generation
- Production SLOs
- Broad compatibility across every model in the vendored `mlx_lm/models/` tree
- Fused Metal kernels
- Large-scale perplexity validation
- Generic Linux or Windows runtime support

## Validation boundary

Two validation layers exist:

1. **Public static checks**
   - packaging metadata
   - source-tree integrity
   - syntax compilation
2. **Local Apple Silicon checks**
   - MLX install
   - unit and integration tests
   - model smoke runs
   - manual memory and latency comparison

Use `scripts/preflight.py` for the first layer and `scripts/validate_apple_silicon.sh` for the second.
