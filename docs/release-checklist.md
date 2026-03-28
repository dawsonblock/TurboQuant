# Release checklist

This is the minimum bar for calling a tagged snapshot technically credible. It is a release gate, not a wish list.

## Static gate

- `python scripts/preflight.py` passes
- `python -m compileall turboquant mlx_lm tests` passes
- `python -m build` produces both sdist and wheel
- `README.md`, `docs/supported-surface.md`, and `docs/validation-local.md` agree on the supported slice

## Apple Silicon gate

Run on an Apple Silicon Mac with MLX installed.

- `pytest tests/unit/` passes
- `pytest tests/integration/` passes
- `./scripts/validate_apple_silicon.sh` passes
- At least one Llama-family smoke run succeeds
- At least one Gemma-family smoke run succeeds
- Dense vs TurboQuant memory numbers are captured and saved with the release notes

## Regression gate

- Non-power-of-two rotation remains orthogonal
- `residual_topk` survives the legacy adapter path
- state save/restore rejects config drift
- deprecated legacy knobs still warn instead of silently changing runtime behavior

## Documentation gate

- No benchmark claim is labeled production unless it is backed by release data
- No CI badge implies MLX runtime certification on generic runners
- Supported models are named explicitly
