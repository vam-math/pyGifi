# PyGifi Tutorial: How to Use

This tutorial demonstrates the practical application of PyGifi for analyzing categorical and mixed-type data using Optimal Scaling techniques.

---

## 1. Getting Started: The `pygifi` API
The Pygifi library adheres strictly to standard Python data science conventions (`scikit-learn` style syntax). The general workflow applies universally to all three primary models (`Homals`, `Princals`, `Morals`):

1.  **Initialize the model:** Choose your dimensions and convergence parameters.
2.  **Fit the model:** Pass your `pandas.DataFrame` or `numpy.ndarray` to `model.fit()`. 
3.  **Inspect the parameters:** Review the calculated optimal quantifications and object scores.
4.  **Visualize:** Generate beautiful Matplotlib visual plots using Pygifi's built-in graphing functions.

## 2. Example 1: `Homals` (Multiple Correspondence Analysis)
`Homals` handles strict nominal variables. Let's analyze a simple dataset. 

```python
import pandas as pd
from pygifi import Homals
from pygifi.data import datasets

# 1. Load a sample dataset
# The "hartigan" dataset contains categorical descriptions of various hardware
df = datasets.get_dataset('hartigan')

# 2. Instantiate Homals
model = Homals(ndim=2, eps=1e-6)

# 3. Fit the model to the data
model.fit(df)

# 4. Access the optimal scaling results
print("Final Stress (Loss):", model.stress_)
print("Eigenvalues (Variance Captured):", model.eigenvalues_)

# You can look at the explicitly calculated optimal quantifications for 'Thread'
print("\nOptimal Quantifications for 'Thread':")
print(model.category_quantifications_['Thread'])

# 5. Visualize the outcome
model.plot_objectscores()  # Shows row data points in the new optimized 2D space
model.plot_quantifications() # Shows how the nominal categories were optimally valued
```

## 3. Example 2: `Princals` (Non-linear PCA)
`Princals` handles mixtures of nominal, ordinal, and continuous variables by enforcing monotone/isotonic regression on variables flagged as "ordinal."

```python
from pygifi import Princals
from pygifi.data import datasets

# 1. Load the "neumann" dataset
df = datasets.get_dataset('neumann')

# 2. Instantiate Princals
# We explicitly define that ALL columns should be treated as ordinal variables
# Ties are maintained strictly ('s'), and missing values get multiple imputation ('m')
model = Princals(ndim=2, ordinal=True, ties='s', missing='m')

# 3. Fit Model
model.fit(df)

# 4. View optimal results
# The output format is identical to Homals, ensuring consistent APIs.
print("Eigenvalues:", model.eigenvalues_)

# 5. Visualization - Transformation Plots
# Since Princals applies monotone regression, we can visualize how the original 
# ordinal integer scales were stretched mathematically to maximize Principal Component variance.
model.plot_transplot(variable='Wind')
```

## 4. Example 3: `Morals` (Optimal Regression)
Instead of extracting variance via PCA, `Morals` attempts to maximize the $R^2$ of a prediction by optimally scaling both `X` and `y`.

```python
import numpy as np
import pandas as pd
from pygifi import Morals

# 1. Generate fake data
np.random.seed(42)
X = pd.DataFrame({
    'Cat1': np.random.choice(['A', 'B', 'C'], 100),
    'Cat2': np.random.choice(['Low', 'Med', 'High'], 100)
})

# Y is generated based on an internal logic where A=10, High=50, etc.
# We want Morals to figure this out without being told the exact metric distance.
y = np.random.rand(100) * 10 

# 2. Instantiate Morals
# We tell Morals that 'Cat2' is ordinal, but 'Cat1' is Nominal
model = Morals(ordinal_x=[False, True], ordinal_y=True)

# 3. Fit 
# Note the standard scikit-learn X, y signature
model.fit(X, y)

# 4. Review prediction accuracy
print("Optimized R^2:", model.r_squared_)
print("Linear Beta Coefficients:", model.beta_)

# 5. Visualize target transformation
model.plot_transplot_y() # Show how the original Y target was non-linearly warped for optimal R^2 fit
```

## 5. Model Saving and Exporting
Because PyGifi strictly implements standard properties, you can export fitted models seamlessly via the `pickle` library, allowing you to reload the model later and execute `model.transform(new_data)` instantly without retraining the optimal scalar mappings.
