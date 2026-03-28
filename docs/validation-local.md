# Local validation on Apple Silicon

Public CI in this repository only checks packaging and syntax. It does **not** certify the MLX runtime path, because GitHub-hosted runners are not Apple Silicon and do not provide a usable `mlx` environment.

Use the local script below on an Apple Silicon Mac for real runtime validation:

```bash
./scripts/validate_apple_silicon.sh
```

That script:

- creates a fresh virtualenv
- installs the package in editable mode with `.[dev,apple]`
- compiles the source tree
- runs the narrow unit and integration tests that exercise the supported TurboQuant path

For manual model smoke tests, run dense generation first, then the TurboQuant upgrade path on the same prompt and compare stability, memory use, and throughput.


You can also invoke the same paths through the Makefile:

```bash
make static-check
make validate-local
```
