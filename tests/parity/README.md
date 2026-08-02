# R ↔ Python parity check

Verifies pygifi's `Homals`, `Princals`, and `Morals` against the real CRAN
`Gifi` package, on every dataset in `pygifi/data/`. These are the only three
models checked because they're the only three that exist in R's `Gifi`
package — there is nothing in R to compare pygifi's other, R-less models
against.

## Files

| File | Role |
|---|---|
| `manifest.json` | Single source of truth: which datasets, which methods, which columns (for Morals' x/y split). Both scripts below read this. |
| `generate_r_ground_truth.R` | Runs real R `Gifi` on every entry in the manifest. Also re-exports each classic dataset to `pygifi/data/*.csv` (the authoritative values R just used). |
| `run_python.py` | Runs pygifi on the same data, using `pygifi.utils.type_inference` to resolve numeric vs. categorical columns automatically. |
| `compare.py` | Pairs up the two output sets, computes % difference per field (with sign-flip tolerance for SVD-derived quantities), classifies PASS (<1%) / WARN (<10%) / FAIL. |
| `plots.py` | One bar chart per method showing max % diff per dataset, colored by verdict. |
| `run_all.py` | Runs all four steps in order. |

`rosenbrock6D.xlsx` (600 rows × 6 predictors) is intentionally excluded from
`manifest.json` — Morals' nested per-iteration isotonic projections make it too
slow for a routine run (minutes, not seconds), not because anything is wrong
with it. Add it back to `morals_runs` if you want that data point too.

## Running it

```
python tests/parity/run_all.py
```

Both R and Python draw their starting object-score matrix from the same
seed (123) via the ported RNG (`pygifi_rng`, built from `pygifi/rng/`) —
R's `gifiEngine()` hardcodes `set.seed(123)` internally, and Python's
`r_seed=123` reproduces the identical matrix independently, so no manual
hand-off of random numbers between the two languages is needed.

## Outputs

```
results/r_ground_truth/<dataset>_<method>.json   -- R output
results/python_output/<dataset>_<method>.json    -- pygifi output
results/comparison/<dataset>_<method>.json       -- per-field % diff + verdict
results/comparison_summary.csv                   -- one row per (dataset, method)
results/plots/<method>_pct_diff.png              -- one chart per method
```
