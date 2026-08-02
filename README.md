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

# Morals uses its own dedicated plot function, not the generic pygifi.plot()
# dispatcher (which is built for Homals/Princals' result structure)
from pygifi.visualization.plot import plot_morals
plot_morals(model.result_)
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

`categorical_encode`/`categorical_decode` work on a single column at a time (a `pd.Series`, list, or array):

```python
import pygifi
import pandas as pd

colors = pd.Series(['red', 'blue', 'red'])

# Encode strings to 1-indexed integer codes
encoded, mapping = pygifi.categorical_encode(colors)

# Decode back to original labels
decoded = pygifi.categorical_decode(encoded, mapping)
```

---

## ⚠️ Important Limitations

- **Homals and Princals do not support out-of-sample projection**, matching R: `model.transform(X)` returns the training-set object scores regardless of what `X` you pass — it does not raise an error, it simply ignores new data. Use `cv_morals()` for cross-validation instead.
- **Morals does support out-of-sample prediction** via `model.predict(X_new)`, which applies the fitted spline/quantification transforms to new data.
- Results match R's Gifi within floating-point tolerance on every dataset covered by the parity suite (see [Validation Against R's Gifi](#️-validation-against-rs-gifi)); two known, explained exceptions are documented in `tests/parity/README.md`.

---

## ⚖️ Validation Against R's Gifi

`tests/parity/` runs pygifi against the real CRAN `Gifi` package on every bundled
dataset and produces a % difference report plus plots.

```bash
python tests/parity/run_all.py
```

This runs, in order:
1. `generate_r_ground_truth.R` — fits `homals`/`princals` on all 12 classic
   datasets and `morals` on 3 regression datasets, using the real R package.
2. `run_python.py` — fits the same models with pygifi on the same data.
3. `compare.py` — pairs up the two outputs field-by-field, computing a
   scale-relative % difference and a PASS (<1%) / WARN (<10%) / FAIL verdict.
4. `plots.py` — one chart per method showing % difference by dataset.

> **Requirements:** R installed with `install.packages("Gifi")`, and (optionally,
> for exact bit-level RNG parity rather than an independent-but-equivalent
> random start) the `pygifi_rng` C extension:
> `cd pygifi/rng && python setup_rng.py build_ext --inplace`

Outputs land in `results/`: `r_ground_truth/`, `python_output/`, and
`comparison/` hold the raw per-(dataset, method) JSON; `comparison_summary.csv`
is the one-row-per-run summary; `plots/` has the charts. See
[`tests/parity/README.md`](tests/parity/README.md) for the full breakdown,
including the two known, explained non-matches (a non-convergence case and a
different-local-optimum case on the largest dataset — both documented there
rather than silently hidden).

---

## 🧪 Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run the full suite
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=pygifi --cov-report=term-missing
```

### Test Fixtures

The `tests/fixtures/` folder contains fixed CSV inputs and R-generated JSON reference outputs used by `test_parity.py`. These fixtures make it possible to compare Python results against saved R `Gifi` outputs in a repeatable way without R installed or rerun for every test case. (This is a different, lighter-weight mechanism than `tests/parity/` above — see `tests/parity/README.md` for how the two relate.)

If the reference fixtures ever need to be regenerated from R, use:

```bash
Rscript tests/fixtures/generate_fixtures.R
```

## 🤝 Community Guidelines

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for how to report bugs, propose changes, run the test suite, and ask for support.

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
│   ├── rng/                        ← C extension for exact R RNG compatibility
│   │   ├── pygifi_rng.c            ← Self-contained port of R's MT19937 + AS241 inversion
│   │   └── setup_rng.py            ← Build script for the extension
│   │
│   ├── visualization/              ← All plotting code
│   │   └── plot.py                 ← Plot dispatcher for all models
│   │
│   └── data/                       ← Built-in datasets
│
├── tests/                          ← Pytest suite (one file per module) + fixtures
│   ├── fixtures/                   ← Frozen R reference outputs used by test_parity.py
│   └── parity/                     ← Live R-vs-Python validation pipeline (see its README.md)
│       ├── manifest.json           ← Datasets/methods covered, single source of truth
│       ├── generate_r_ground_truth.R
│       ├── run_python.py
│       ├── compare.py
│       ├── plots.py
│       └── run_all.py              ← Runs all of the above in order
│
├── results/                        ← Output of tests/parity/run_all.py
├── setup.py / pyproject.toml       ← Package configuration
└── README.md                       ← This file
```


---

## 📄 License

GPL-3.0-or-later — same as the original R Gifi package.

## 🙏 Credits

Original R Gifi package by Patrick Mair, Jan de Leeuw, and Patrick Groenen. Python port developed for research reproducibility.
The R to python conversion was developed by Vamanie Perumal, Indian Institute of Technology Madras and Sanjeev, Bhavesh from Amrita College of Engineering, Chennai.
