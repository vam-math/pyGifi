# pygifi — Python Port of R's Gifi Library 🐍📊

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL%203.0-green.svg)](https://opensource.org/licenses/GPL-3.0)
[![CI](https://github.com/vam-math/pyGifi/actions/workflows/ci.yml/badge.svg)](https://github.com/vam-math/pyGifi/actions/workflows/ci.yml)

**pygifi** is a Python port of the R [Gifi package](https://cran.r-project.org/package=Gifi) by Mair, De Leeuw, and Groenen.  
It brings **multivariate analysis with optimal scaling** to Python — handling categorical, ordinal, and mixed-type data natively in a scikit-learn compatible API.

---

## ✨ What Does This Library Do?

Gifi methods are a family of algorithms that find the best numerical representation of any kind of data — even if it's categorical or ordinal — so you can apply linear methods like PCA or regression to it. Each variable is **optimally transformed** to maximize structure.

| Method | Class | What it does |
|--------|-------|-------------|
| Homogeneity Analysis | `Homals` | Like Multiple Correspondence Analysis (MCA). Finds groups and patterns in categorical data. |
| Optimal Scaling PCA | `Princals` | Like PCA but works on any mix of nominal, ordinal, and numeric variables. |
| Monotone Regression | `Morals` | Like linear regression but the predictors/response are optimally transformed (monotone). |
| Correlational Analysis | `Corals` | Maximizes correlation between two sets of variables. |
| Canonical Analysis | `Canals` | Canonical correlation with optimal scaling. |
| Discriminant Analysis | `Criminals` | Nonlinear discriminant analysis. |
| Multiset Analysis | `Overals` | Generalizes Homals/Corals to multiple sets. |
| Primal Analysis | `Primals` | Regression with metric response. |
| Additive Analysis | `Addals` | Additive models with optimal scaling. |
| Missing Data Imputation | `GifiIterativeImputer` | Iterative imputation oriented to the Gifi framework. |

**Additional utilities:**
- `pygifi.plot()` — unified plot dispatcher (loading plots, biplots, transformation plots, object score plots)
- `pygifi.get_dataset()` — 12 built-in classic datasets
- `pygifi.make_numeric()` / `encode()` / `decode()` — categorical coding utilities
- `pygifi.cv_morals()` — cross-validation for Morals
- `pygifi.knots_gifi()` — B-spline knot placement (matches R exactly)

---

## 📦 Installation

### Requirements

- Python **3.9 or above**
- pip

### Install from source (recommended until PyPI release)

```bash
# 1. Clone the repository
git clone https://github.com/vam-math/pyGifi.git
cd pyGifi

# 2. (Optional but recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install the package with all dependencies
pip install -e .
```

The `-e` flag installs in **editable mode**, so any local file changes are reflected immediately without reinstalling.

### Dependencies installed automatically

| Package | Purpose |
|---------|---------|
| `numpy >= 1.21` | Numerical computation |
| `scipy >= 1.7` | Linear algebra, splines, NNLS |
| `pandas >= 1.3` | DataFrame handling |
| `scikit-learn >= 1.0` | Base estimator interface |
| `matplotlib >= 3.4` | Plotting |

### Optional: Numba acceleration

For faster PAVA (isotone regression) on large datasets:

```bash
pip install -e ".[accelerate]"
```

### Verify installation

```bash
python -c "import pygifi; print(pygifi.__version__)"
```

---

## 🚀 Quick Start

### 1. Homogeneity Analysis (Homals)

Discover structure in categorical data:

```python
import pygifi

# Load a built-in dataset (12 classic datasets available)
df = pygifi.get_dataset('ABC')

# Fit Homals — 2 dimensions, treat all columns as nominal
model = pygifi.Homals(ndim=2, levels='nominal')
model.fit(df)

# Inspect results
print(model)                              # summary
print(model.result_['objectscores'][:5]) # row coordinates
print(model.result_['evals'])            # eigenvalues

# Plot
pygifi.plot(model, plot_type='objplot')  # object scores
pygifi.plot(model, plot_type='loadplot') # variable loadings
```

### 2. Optimal Scaling PCA (Princals)

Like PCA for mixed-type data:

```python
import pygifi

df = pygifi.get_dataset('galo')

model = pygifi.Princals(ndim=2, levels=['nominal', 'ordinal', 'nominal', 'ordinal'])
model.fit(df)
print(model)
pygifi.plot(model, plot_type='biplot')
```

### 3. Monotone Regression (Morals)

Regression where variables are monotonically transformed:

```python
import pygifi
import pandas as pd

df = pygifi.get_dataset('neumann')
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

model = pygifi.Morals(xdegrees=2, ydegrees=2, xordinal=True, yordinal=True)
model.fit(X, y)
print(model)                        # SMC, loss, iterations
pygifi.plot(model, plot_type='transplot')
```

### 4. Available Built-in Datasets

```python
import pygifi

# See all available datasets
datasets = ['ABC', 'galo', 'hartigan', 'neumann', 'mammals',
            'roskam', 'senate07', 'gubell', 'house', 'sleeping',
            'small', 'WilPat2']

df = pygifi.get_dataset('hartigan')
print(df.head())
```

### 5. Categorical Encoding Utilities

```python
import pygifi
import pandas as pd

df = pd.DataFrame({'color': ['red', 'blue', 'red'], 'size': ['S', 'L', 'M']})

# Encode strings to integer codes
encoded, mapping = pygifi.categorical_encode(df)

# Decode back to original
decoded = pygifi.categorical_decode(encoded, mapping)
```

---

## ⚠️ Important Limitations

- **Out-of-sample prediction is not supported.** The optimal transformations are computed on the entire dataset at once. Calling `model.transform(X_test)` on unseen data will raise `NotImplementedError`. Use `cv_morals()` for cross-validation instead.
- Results match R's Gifi within floating-point tolerance. Large datasets may show minor differences due to RNG initialization differences.

---

## ⚖️ Validation Against R's Gifi

This project includes a **3-Phase Automated Validation Suite** that runs both the Python and R Gifi implementations end-to-end, producing extensive comparison reports and plots.

### One-command run (from the project root)

```bash
python3 compare_test.py
```

This single command executes all three phases:
1. **Phase 1: Numerical Accuracy** — Preprocesses datasets, runs PyGifi and R Gifi locally, and performs a category-by-category diff on model outputs tracking discrepancies.
2. **Phase 2: Distribution Comparison** — Analyzes differences in empirical skewness, kurtosis, and overlays histograms for visual verification inside `validation/results/plots/`.
3. **Phase 3: Structural PCA Comparison** — Evaluates macro-structural differences via Eigenvalues, Object Scores, and Component Loadings scatter plots.

> **Requirements:** 
> - R must be installed with `install.packages("Gifi")`.
> - **Exact Parity (1e-6)** requires compiling the `pygifi_rng` C extension representing R's random number generator initial state matrix setup:
>   `cd pygifi/rng && python3 setup_rng.py build_ext --inplace`

### Running individual master test scripts

You can also run either side independently to inspect full model output (eigenvalues, loadings, category quantifications, and transformed dataset) for every dataset:

```bash
# Python side — loops over all datasets in validation/datasets/processed/
python3 pyGifi_test.py
# Output: console + validation/results/python_master_report.txt

# R side — same loop
Rscript Gifi_test.R
# Output: console + validation/results/r_master_report.txt
```

Both scripts automatically detect all datasets in `validation/datasets/processed/` — no hardcoded paths.

### Adding your own dataset

1. Drop your CSV into `validation/datasets/`
2. Run `python3 compare_test.py` — it will be picked up automatically.

---

## 🧪 Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests (excluding slow R-parity tests)
pytest tests/ --ignore=tests/test_parity.py -v

# Run with coverage
pytest tests/ --ignore=tests/test_parity.py --cov=pygifi --cov-report=term-missing
```

## 🤝 Community Guidelines

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for how to report bugs, propose changes, run the test suite, and ask for support.

---

## 📊 Analysis / Visualization

We provide tools for generating comparison plots against R, as well as standalone batch analysis tools.

### R Comparison Plots

Generate three types of comparative plots (`01_raw_distributions.png`, `02_trend_plots.png`, `03_pca_plots.png`) using:

```bash
python3 plot.py
```

> **Requirements:** R with `install.packages("Gifi")`. If R is unavailable, plots are generated for Raw and PyGifi only.

### Batch Model Analysis (master_analysis.py)

If you just want to run PyGifi models (`Homals`, `Princals`, `Morals`) on multiple datasets and automatically generate visualization plots (Biplots, Transplots, Screeplots) without concerning yourself with R comparisons:

```bash
python3 master_analysis.py
```
This script loops over all datasets in the project, fits corresponding Gifi models, and drops all output plots directly into `results/master_analysis/plots/`.

---

## 📁 Project Structure — Explained for Beginners

```text
pyGifi/
│
├── pygifi/                         ← The actual Python library (install this)
│   ├── __init__.py                 ← Entry point: exposes all public classes/functions
│   │
│   ├── models/                     ← One file per Gifi algorithm
│   │   ├── homals.py               ← Homals: homogeneity analysis (MCA-style)
│   │   ├── princals.py             ← Princals: optimal scaling PCA
│   │   ├── morals.py               ← Morals: monotone regression
│   │   ├── corals.py               ← Corals: correlation analysis
│   │   ├── canals.py               ← Canals: canonical analysis
│   │   ├── criminals.py            ← Criminals: discriminant analysis
│   │   ├── overals.py              ← Overals: multi-set analysis
│   │   ├── primals.py              ← Primals: metric regression
│   │   ├── addals.py               ← Addals: additive analysis
│   │   └── impute.py               ← GifiIterativeImputer: missing value imputation
│   │
│   ├── core/                       ← Engine internals
│   │   ├── engine.py               ← Main ALS loop + transformation routing
│   │   ├── structures.py           ← Data structure builders
│   │   ├── linalg.py               ← Linear algebra helpers
│   │   └── cv.py                   ← Cross-validation: cv_morals()
│   │
│   ├── utils/                      ← Low-level utilities (internal helpers)
│   │   ├── _cone.py                ← Cone projection router
│   │   ├── isotone.py              ← PAVA, Dykstra, monotone regression functions
│   │   ├── splines.py              ← B-spline basis construction
│   │   ├── coding.py               ← Categorical coding, encoding/decoding
│   │   ├── utilities.py            ← Matrix manipulation components
│   │   └── prepspline.py           ← Spline knot pre-processing utilities
│   │
│   ├── rng/                        ← C Extension for exact R compatibility
│   │   ├── rng.c                   ← R's Mersenne-Twister / Normal generation ported to C
│   │   ├── pygifi_rng.c            ← Python/C API Wrapper
│   │   └── setup_rng.py            ← Build script for the extension
│   │
│   ├── visualization/              ← All plotting code
│   │   └── plot.py                 ← Plot dispatcher for all models
│   │
│   └── data/                       ← Built-in datasets
│
├── validation/                     ← Automated 3-Phase R vs Python Validation Suite
│   ├── datasets/                   
│   │   ├── processed/              ← Pre-processed CSVs ready for modeling
│   │   └── final_dataset/          ← Organized final transformed model outputs
│   ├── python_scripts/
│   │   └── run_pygifi.py           ← PyGifi model execution script
│   ├── r_scripts/
│   │   └── run_gifi.R              ← R implementation script
│   ├── compare/                    
│   │   ├── compare_results.py      ← Phase 1: Numerical Accuracy comparison
│   │   ├── visualize_distributions.py ← Phase 2: Distribution Comparison
│   │   └── pca_comparison.py       ← Phase 3: Structural PCA Difference calculation
│   ├── results/                    ← Summary comparison CSVs and parameter diff txts
│   ├── preprocess_datasets.py      ← Cleaning raw datasets step
│   └── report.py                   ← Orchestration of Phases 1 to 3
│
├── compare_test.py                 ← Launcher: Runs the 3-Phase validation pipeline
├── master_analysis.py              ← Batch Tool: Runs Homals/Princals/Morals on datasets + plots
├── pyGifi_test.py                  ← Standalone test: Print PyGifi outputs
├── Gifi_test.R                     ← Standalone test: Print R Gifi outputs
├── diag_gifi.R                     ← Diagnostics test inside R directly
├── plot.py                         ← Compare dataset structures via auto-generated R overlays
│
├── analysis/                       ← Visualization analysis
│   ├── analyze.py                  ← Main analysis script
│   ├── get_r_transforms.R          ← R helper: runs Gifi Princals
│   ├── r_transforms/               ← Auto-generated R transformed CSVs
│   └── plots/                      ← All generated plots
│
├── docs/                           ← Documentation and Jupyter Notebook tutorials
├── examples/                       ← Standalone worked code examples
├── tests/                          ← Automated Pytest Suite (ignores slow parity by default)
├── setup.py / pyproject.toml       ← Package configurations
└── README.md                       ← This file
```


---

## 📄 License

GPL-3.0-or-later — same as the original R Gifi package.

## 🙏 Credits

Original R Gifi package by Patrick Mair, Jan de Leeuw, and Patrick Groenen. Python port developed for research reproducibility.
The R to python conversion was developed by Sanjeev, Bhavesh from Amrita College of Engineering, Chennai and Vamanie Perumal, Indian Institute of Technology Madras.
