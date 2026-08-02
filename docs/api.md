# PyGifi API Reference

The PyGifi library provides three main classes: `Homals`, `Princals`, and `Morals`. They inherit standard `scikit-learn` conventions (i.e. `fit`, `transform`, `fit_transform`) but have significant specialized attributes for optimal scaling.

---

## 1. Homals

Homogeneity Analysis by Alternating Least Squares (Multiple Correspondence Analysis).

```python
from pygifi import Homals
model = Homals(ndim=2, eps=1e-6, itmax=1000)
```

### Parameters
*   **ndim**: `int`, default=`2`. The number of dimensions to fit.
*   **eps**: `float`, default=`1e-6`. Convergence tolerance threshold. Iterations stop when the absolute change in the loss function `f` is smaller than `eps`.
*   **itmax**: `int`, default=`1000`. Maximum number of ALS iterations allowed.

### Methods
*   **`fit(X)`**: Fit the Homals model to the data matrix `X` (Nominal data only). Handles string or integer categories automatically.
*   **`transform(X)`**: Applies the calculated optimal quantifications to new data or the original data, projecting it into the fitted `ndim` space. Returns the object scores.
*   **`fit_transform(X)`**: Fits the model and immediately applies the projection.
*   **`plot_objectscores()`**: Generates a 2D scatter plot of the fitted object scores.
*   **`plot_quantifications(variable=None)`**: Generates a plot of the optimal category scale assigned (optionally filtered by a specific `variable` name).

### Attributes
After calling `fit()`, the following attributes are available:
*   `model.objectscores_`: `np.ndarray` of shape `(n_samples, ndim)`. The row coordinates.
*   `model.category_quantifications_`: `dict`. Maps each variable name to a `pd.DataFrame` containing the optimal multivariate scaling values for each category level.
*   `model.eigenvalues_`: `np.ndarray`. The variances captured along each of the `ndim` dimensions.
*   `model.loadings_`: `pd.DataFrame`. The column coordinates / variable relationships.
*   `model.stress_`: `float`. The final optimized value of the internal cost function.
*   `model.iters_`: `int`. The number of iterations executed before convergence.

---

## 2. Princals

Principal Components Analysis by Alternating Least Squares (Non-linear PCA).

```python
from pygifi import Princals
model = Princals(ndim=2, ordinal=True, ties='s', missing='m', eps=1e-6, itmax=1000)
```

### Parameters
*   **ndim**: `int`, default=`2`. Number of dimensions.
*   **ordinal**: `bool` or `list`, default=`True`. If `True`, enforces isotonic/monotone regression (order-preserving scaling) on *all* variables. If a `list` of booleans or strings, applies ordinal scaling selectively per column. `False` degenerates the column to purely nominal scaling.
*   **ties**: `str` or `list`, default=`'s'`. Tie-handling metric for PAVA ordinal scaling. `'s'` implies secondary tie-breaking (tied values must remain exactly tied after scaling), `'p'` is primary (tied values can diverge if it improves fit), `'t'` is tertiary.
*   **missing**: `str` or `list`, default=`'m'`. Missing value methodology. `'m'` signifies Multiple Imputation via Alternating Least Squares. `'s'` signifies Single Imputation, treating `NaN` as a distinct passive category.
*   **eps**: `float`, default=`1e-6`.
*   **itmax**: `int`, default=`1000`.

### Methods
*   Similar to `Homals`: `fit()`, `transform()`, `fit_transform()`.
*   **`plot_loadings()`**: Generates a vector plot in `ndim` space for the component matrix.
*   **`plot_transplot(variable=None)`**: Plots the non-linear transformation curve (Original Values vs. Optimally Scaled Values) for a given variable.

### Attributes
Matches `Homals` with identical geometric spaces.

---

## 3. Morals

Multiple Optimal Regression Iterating Alternating Least Squares. Fits the model `Y = B * X + \epsilon`.

```python
from pygifi import Morals
model = Morals(ordinal_y=True, ordinal_x=True, ties='s', missing='m', eps=1e-6, itmax=1000)
```

### Parameters
*   **ordinal_y**: `bool`, default=`True`. Enforce monotonic optimal scaling on the target variable.
*   **ordinal_x**: `bool` or `list`, default=`True`. Enforce monotonic optimal scaling on the predictor matrix (`X`).
*   Remaining parameters match `Princals`.

### Methods
*   **`fit(X, y)`**: Fit the non-linear regression mapping predicting `y` from `X`.
*   **`predict(X)`**: Returns the predicted optimized `y` values.
*   **`plot_transplot(variable=None)`**: Show non-linear transformations for a specific predictor.
*   **`plot_transplot_y()`**: Show the non-linear scaling applied to the target variable `y`.

### Attributes
*   `model.beta_`: `np.ndarray`. The finalized linear regression coefficients applied to the *optimally scaled* versions of `X`.
*   `model.r_squared_`: `float`. The traditional Multiple Correlation Coefficient $R^2$ of the optimally scaled model.
*   `model.y_hat_`: `np.ndarray`. The in-sample predicted scaled target values.
*   `model.y_res_`: `np.ndarray`. Regression residuals.
*   `model.x_hat_`: `np.ndarray`. The continuous optimally scaled matrix for `X`.
