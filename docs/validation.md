# PyGifi Validation and Benchmark System

To ensure mathematical correctness matching exactly with the R CRAN `Gifi` package across different operating systems, `PyGifi` contains a complete benchmarking framework in `validation/`.

## Architecture of Validation
1. **r_scripts/**: Re-computes results across multi-dimensional benchmarks natively using installed R instances, guaranteeing strict source of truth limits.
2. **python_scripts/**: Computes the same iterations identically inside Python using `PyGifi`.
3. **compare/**: Assesses output variance matrices utilizing an acceptable numeric tolerance threshold (`tol=1e-3`) avoiding false positives from diverse BLAS linear algebra packages.

## Running Parity Benchmarks

If you have R correctly sourced in your path (`Rscript`), our `pytest` wrapper automatically incorporates it:

```bash
pytest tests/test_r_parity_full.py
```

Alternatively, invoke the report console directly from validation rules folder:

```bash
python validation/report.py
```
