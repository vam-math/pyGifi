# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Fixed
- `Morals` used the wrong default spline knot configuration (`type="E"`, zero
  interior knots) for predictor and response transforms, causing beta
  coefficients to diverge from R by 1-2%. Now matches R's actual default
  (`knotsGifi(x, "Q")`, quantile knots with 3 interior knots) exactly
  (verified to ~1e-11 against live R execution).
- `gifi_majorization`/`gifi_als` violated their documented "stress never
  increases" guarantee: the `Z_j` update omitted the `(A_j A_j^T)^-1` term
  required for a bilinear least-squares solution, the object-score update
  used a scalar Frobenius rescale instead of the exact orthogonal-Procrustes
  projection, and ordinal (PAVA) updates used unweighted pooling on
  unequal-sized categories instead of weighting by category count. All three
  fixed; monotonicity is now exact rather than approximate.
- `cor_list` silently assumed pre-centered, unit-normalized input columns;
  it now centers/normalizes internally so it computes a true correlation
  matrix for any input, matching its documented contract.

### Removed
- `Overals`, `Canals`, `Addals`, `Corals`, `Criminals`, `Primals`: none have
  an equivalent in the real CRAN `Gifi` package (verified against its
  exported function list), so there was no way to validate them against a
  reference implementation.

### Added
- Real R-vs-Python parity pipeline (`tests/parity/`) validating `Homals`,
  `Princals`, and `Morals` against the actual CRAN `Gifi` package across all
  12 bundled datasets, with a scale-relative % difference report and plots
  (see `results/`).
- `pygifi_rng` C extension (`pygifi/rng/`) now builds cleanly from a single
  self-contained source file and is verified bit-exact against R's
  `set.seed()`/`rnorm()` output (`tests/test_rng_parity.py`).

## [0.1.0] - 2026-03-08

### Added
- Initial release: Python port of R Gifi library
- `Homals`: Multiple Correspondence Analysis
- `Princals`: Categorical Principal Component Analysis
- `Morals`: Monotone Regression Analysis
- `pygifi.plot`: Visualization utilities (plot_homals, plot_princals, plot_morals)
- Core primitives: Modified Gram-Schmidt, PAVA, B-spline basis
- scikit-learn compatible API (fit/transform)
