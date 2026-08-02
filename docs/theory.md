# PyGifi Theory and Mathematical Foundation

**PyGifi** is a Python library for Multivariate Analysis with Optimal Scaling. It is a direct mathematical port of the beloved CRAN `Gifi` package written by Jan de Leeuw and Patrick Mair. 

The term "Gifi" originates from Albert Gifi, a pseudonymous collective of Dutch statisticians (including De Leeuw, Meulman, etc.) who conceptualized a unified framework for categorical data analysis through alternating least squares.

---

## 1. Optimal Scaling
Optimal scaling resolves a core issue in traditional linear multivariate statistics: handling categorical and ordinal variables properly. Standard PCA expects continuous metric data. If you feed it nominal (unordered categories) or ordinal (ordered categories) data encoded as integers (e.g., Small=1, Medium=2, Large=3), you impose arbitrary linear distances between these categories.

**Optimal Scaling** replaces these arbitrary integers with mathematically derived "category quantifications." It searches for the optimal numeric values for each category that maximize the variance explained by the underlying model (like PCA) while strictly respecting the variable's original scaling level:

*   **Nominal**: Categories can be assigned *any* distinct values. Ordering is ignored.
*   **Ordinal**: Categories must maintain their rank order, but distances can stretch or shrink (Isotonic/Monotone Regression).
*   **Metric**: Categories retain their original linear intervals (equivalent to standard PCA).
*   **Spline**: Categories are smoothed using B-splines of a specific degree, allowing non-linear but smooth transformations.

## 2. Alternating Least Squares (ALS)
At the heart of Gifi is the **Alternating Least Squares (ALS)** algorithm. The Gifi system defines a global loss function (often called "Stress") that measures the discrepancy between the observed categorical data and a low-dimensional target space (usually defined by "Object Scores" and "Component Loadings").

The ALS algorithm iteratively minimizes this loss function in two alternating steps until convergence:

1.  **Model Update**: Fix the data transformations (the optimal scaling quantifications) and update the underlying geometric model parameters (e.g., the object scores and component loadings, often via SVD or Gram-Schmidt orthogonalization).
2.  **Optimal Scaling Update**: Fix the geometric model parameters and update the data transformations. For each variable, calculate a "target" projection and then project that target onto the constraint cone defined by the variable's restriction level (e.g., pool-adjacent-violators for ordinal constraints, or simple averaging for nominals).

Because both steps strictly minimize the same global loss function, the algorithm is guaranteed to converge monotonically.

## 3. Majorization
In cases where strict orthogonal rotation constraints are required outside standard ALS flows (such as non-linear PCA models like `Princals`), PyGifi utilizes **Majorization** algorithms (specifically, Smacof-style multiplicative iterative updates). 

Majorization creates a surrogate mathematical matrix (the "majorizing" function) that touches the original cost function at the current estimate but is otherwise strictly larger. By minimizing this simpler quadratic surrogate, the original complex loss function is inherently reduced.

## 4. Models defined in PyGifi

### Homals (Multiple Correspondence Analysis - MCA)
`Homals` (Homogeneity Analysis by Alternating Least Squares) is the Gifi equivalent of Multiple Correspondence Analysis (MCA). It operates exclusively on nominal data. It seeks to assign scores to objects (rows) and categories (levels) such that the dispersion of objects within a category is minimal, while the dispersion between categories is maximal.

### Princals (Non-linear Principal Component Analysis)
`Princals` (Principal Components Analysis by Alternating Least Squares) generalizes standard PCA. While standard PCA only supports metric (linear) variables, `Princals` supports mixed variables. You can simultaneously analyze metric, ordinal, and nominal variables, fitting them into the same lower-dimensional coordinate space.

### Morals (Multiple Optimal Regression Iterating Alternating Least Squares)
`Morals` is a framework for Non-linear Multiple Regression. Instead of assuming a strictly linear relationship $Y = \beta X$, `Morals` allows both the predictor variables ($X$) and the target variable ($Y$) to be optimally scaled. It finds the optimal non-linear transformations of all variables that maximize the standard $R^2$ fit of the regression equation.
