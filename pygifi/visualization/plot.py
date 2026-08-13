# mypy: ignore-errors
"""
pygifi.plot: Visualization utilities for Gifi method results.

Python port of Gifi/R/plot.homals.R, Gifi/R/plot.princals.R, Gifi/R/plot.morals.R
(Mair, De Leeuw, Groenen. GPL-3.0).

Functions
---------
plot_homals    : Plot object scores and category quantifications (homals result)
plot_princals  : Biplot of object scores and component loadings (princals result)
plot_morals    : Transformation and fitted-vs-residual plot (morals result)
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_homals(result, dim=None, which='objectscores', ax=None, **kwargs):
    """
    Plot homals results: object scores and/or category quantifications.

    Python port of R's plot.homals().

    Parameters
    ----------
    result : dict or object with attributes
        Output from Homals.fit().
    dim : list of 2 ints, default=[0, 1]
        Indices of dimensions to plot on x and y axes.
    type : str, optional
        Type of plot. One of 'jointplot', 'objplot', 'biplot', 'screeplot', 'transplot'.
        Defaults to 'jointplot'.
    which : str or list, optional
        Deprecated mapping parameter. Kept for backwards compatibility.
    ax : matplotlib.axes.Axes, optional
        Axes to plot into. If None, a new figure is created.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if dim is None:
        dim = [0, 1]
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    # Handle both dict and object-style results
    def get(r, k):
        return r[k] if isinstance(r, dict) else getattr(r, k)

    plot_type = kwargs.pop('type', 'jointplot')
    # Fallback to older `which` parameter
    if which != 'objectscores':
        plot_type = which

    if plot_type == 'screeplot':
        evals = np.asarray(get(result, 'evals'))
        ax.bar(np.arange(len(evals)) + 1, evals, color='steelblue', alpha=0.7)
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Eigenvalue')
        ax.set_title('Homals: Scree Plot')
        ax.set_xticks(np.arange(len(evals)) + 1)
        return ax

    if plot_type == 'transplot':
        # Transformation plot groups by variable
        transforms = get(result, 'transform')
        data = get(result, 'data')
        var_colors = plt.cm.Set1(np.linspace(0, 1, len(transforms)))
        for j, tr in enumerate(transforms):
            # plot first copy of transform against original numeric values
            col_data = np.asarray(get(result, 'datanum')[:, j])
            sorted_idx = np.argsort(col_data)
            ax.step(
                col_data[sorted_idx],
                np.asarray(tr)[
                    sorted_idx,
                    0],
                where='mid',
                color=var_colors[j],
                label=data.columns[j] if hasattr(
                    data,
                    'columns') else f"Var{j}",
                alpha=0.8)

        ax.set_xlabel('Original Values (quantified)')
        ax.set_ylabel('Transformed Scale')
        ax.set_title('Homals: Transformation Plot')
        ax.legend()
        return ax

    which_list = ['objectscores', 'quantifications'] if plot_type == 'jointplot' else [
        'objectscores'] if plot_type == 'objplot' else ['objectscores', 'loadings']

    if 'objectscores' in which_list:
        scores = np.asarray(get(result, 'objectscores'))
        ax.scatter(scores[:, dim[0]], scores[:, dim[1]],
                   color='steelblue', alpha=0.4, s=8, zorder=2, **kwargs)

    if 'quantifications' in which_list:
        # list or dict of (n_cats, ndim) arrays
        quants = get(result, 'quantifications')
        data = get(result, 'data')                # original DataFrame
        var_colors = plt.cm.Set1(np.linspace(0, 1, len(quants)))

        # Normalize quantifications to an iterable of arrays
        quants_list = list(
            quants.values()) if isinstance(
            quants,
            dict) else quants

        for j, q in enumerate(quants_list):
            q = np.asarray(q)
            # Get category labels from the original data column
            col = data.iloc[:, j]
            if hasattr(col, 'cat'):
                cat_labels = col.cat.categories.tolist()
            else:
                cat_labels = sorted(col.unique())

            # Handle 1D quantifications (variables with <= ndim categories)
            if q.ndim == 1:
                q = q[:, None]
            if q.shape[1] <= dim[1]:
                # Pad with zeros for missing dimensions
                padded = np.zeros((q.shape[0], dim[1] + 1))
                padded[:, :q.shape[1]] = q
                q = padded

            n_cats = q.shape[0]
            for i in range(min(n_cats, len(cat_labels))):
                ax.scatter(q[i, dim[0]], q[i, dim[1]],
                           color=var_colors[j], marker='^', s=60,
                           edgecolors='black', linewidths=0.5, zorder=5)
                ax.annotate(str(cat_labels[i]),
                            xy=(q[i, dim[0]], q[i, dim[1]]),
                            xytext=(4, 4), textcoords='offset points',
                            fontsize=7, color=var_colors[j], fontweight='bold',
                            bbox=dict(facecolor='white', alpha=0.7,
                                      edgecolor='none', pad=0.3))

    if 'loadings' in which_list:
        try:
            loadings = np.asarray(get(result, 'loadings'))
            scores = np.asarray(get(result, 'objectscores'))
            scalef = np.percentile(np.abs(scores), 80) / \
                (np.max(np.abs(loadings)) + 1e-12)
            for i in range(loadings.shape[0]):
                ax.annotate('',
                            xy=(loadings[i,
                                         dim[0]] * scalef,
                                loadings[i,
                                         dim[1]] * scalef),
                            xytext=(0,
                                    0),
                            arrowprops=dict(arrowstyle='->',
                                            color='coral',
                                            lw=1.5))
        except Exception:
            pass

    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel(f'Dimension {dim[0] + 1}')
    ax.set_ylabel(f'Dimension {dim[1] + 1}')
    ax.set_title('Homals: Object Scores & Quantifications')

    return ax


def plot_princals(result, dim=None, type='biplot', ax=None, **kwargs):
    """
    Plot princals results: biplot of object scores and loadings.

    Python port of R's plot.princals().

    Parameters
    ----------
    result : dict or object
        Output from Princals.fit().
    dim : list of 2 ints, default=[0, 1]
        Dimensions to plot.
    type : str
        'biplot': overlay object scores and loadings.
        'scores': only object scores.
        'loadings': only component loadings (bar chart).
        'screeplot': eigenvalue bar chart.
        'transplot': original vs transformed data.
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    if dim is None:
        dim = [0, 1]

    def get(r, k):
        return r[k] if isinstance(r, dict) else getattr(r, k)

    if type == 'screeplot':
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 5))
        evals = np.asarray(get(result, 'evals'))
        ax.bar(np.arange(len(evals)) + 1, evals, color='steelblue', alpha=0.7)
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Eigenvalue')
        ax.set_title('Princals: Scree Plot')
        ax.set_xticks(np.arange(len(evals)) + 1)
        return ax

    if type == 'transplot':
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 6))
        transforms = get(result, 'transform')
        data = get(result, 'data')

        # If transform is a list of matrices, handle it. If it's a single
        # matrix, split by columns.
        if isinstance(transforms, np.ndarray):
            transforms = [transforms[:, i:i + 1]
                          for i in range(transforms.shape[1])]

        var_colors = plt.cm.Set1(np.linspace(0, 1, len(transforms)))
        for j, tr in enumerate(transforms):
            col_data = np.asarray(get(result, 'datanum')[:, j])
            sorted_idx = np.argsort(col_data)
            ax.step(
                col_data[sorted_idx],
                np.asarray(tr)[
                    sorted_idx,
                    0],
                where='mid',
                color=var_colors[j],
                label=data.columns[j] if hasattr(
                    data,
                    'columns') else f"Var{j}",
                alpha=0.8)

        ax.set_xlabel('Original Values (quantified)')
        ax.set_ylabel('Transformed Scale')
        ax.set_title('Princals: Transformation Plot')
        ax.legend()
        return ax

    if type == 'loadings':
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        loadings = np.asarray(get(result, 'loadings'))
        n_vars = loadings.shape[0] if loadings.ndim > 1 else 1
        x_pos = np.arange(n_vars)
        ax.bar(x_pos, loadings[:, dim[0]] if loadings.ndim > 1 else loadings,
               color='steelblue', alpha=0.7, label=f'Dim {dim[0] + 1}')
        if loadings.ndim > 1 and loadings.shape[1] > 1:
            ax.bar(x_pos, loadings[:, dim[1]], color='coral', alpha=0.7,
                   label=f'Dim {dim[1] + 1}')
        ax.set_title('Princals: Component Loadings')
        ax.legend()
    else:
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 6))
        scores = np.asarray(get(result, 'objectscores'))
        ax.scatter(scores[:,
                          dim[0]],
                   scores[:,
                   dim[1]],
                   color='steelblue',
                   alpha=0.6,
                   zorder=3,
                   label='Objects',
                   **kwargs)

        if type == 'biplot':
            try:
                loadings = np.asarray(get(result, 'loadings'))
                scalef = np.percentile(np.abs(scores),
                                       80) / (np.max(np.abs(loadings)) + 1e-12)
                for i in range(loadings.shape[0]):
                    ax.annotate('',
                                xy=(loadings[i,
                                             dim[0]] * scalef,
                                    loadings[i,
                                             dim[1]] * scalef),
                                xytext=(0,
                                        0),
                                arrowprops=dict(arrowstyle='->',
                                                color='coral',
                                                lw=1.5))
            except Exception:
                pass  # Silently skip loadings overlay if not available

        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.set_xlabel(f'Dimension {dim[0] + 1}')
        ax.set_ylabel(f'Dimension {dim[1] + 1}')
        ax.set_title('Princals: Biplot')

    return ax


def plot_morals(result, ncols=2):
    """Mirrors R's plot.morals: observed vs transformed per variable."""
    def get(r, k):
        return r[k] if isinstance(r, dict) else getattr(r, k)

    # support both Morals object and dict containing results
    data_X = get(
        result,
        'X_') if hasattr(
        result,
        'X_') else (
            result.get('data') if isinstance(
                result,
                dict) else getattr(
                    result,
                    'data',
                None))
    if data_X is None or not hasattr(data_X, 'columns'):
        # fallback if attributes not found
        num_preds = result.get('n_pred_') if isinstance(
            result, dict) else getattr(
            result, 'n_pred_', None)
        xhat_matrix = result.get('xhat') if isinstance(
            result, dict) else getattr(
            result, 'xhat', getattr(
                result, 'result_', {}).get(
                'xhat', np.empty(
                    (0, 1))))
        num_preds = num_preds or (
            xhat_matrix.shape[1] if xhat_matrix is not None else 1)
        cols = [f"X{i}" for i in range(num_preds)]
        X_obs = np.asarray(data_X) if data_X is not None else np.zeros(
            (xhat_matrix.shape[0], len(cols)))
    else:
        cols = list(data_X.columns)
        X_obs = data_X.values

    y_obs = result.get('y_') if isinstance(
        result, dict) else getattr(
        result, 'y_', None)
    if y_obs is None:
        y_obs = np.zeros(len(X_obs))

    xhat = getattr(
        result,
        'result_',
        {}).get('xhat') if hasattr(
        result,
        'result_') else (
            result.get('xhat') if isinstance(
                result,
                dict) else getattr(
                    result,
                    'xhat',
                    np.empty(
                        (0,
                         1))))
    yhat = getattr(
        result,
        'result_',
        {}).get('yhat') if hasattr(
        result,
        'result_') else (
            result.get('yhat') if isinstance(
                result,
                dict) else getattr(
                    result,
                    'yhat',
                    np.zeros(
                        len(X_obs))))

    nplots = len(cols) + 1   # predictors + response
    nrows = int(np.ceil(nplots / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()

    # predictor transformations
    for i, col in enumerate(cols):
        x_obs_i = X_obs[:, i]
        x_hat_i = xhat[:, i]   # transformed predictor i
        order = np.argsort(x_obs_i)
        axes[i].plot(x_obs_i[order], x_hat_i[order], 'k-')
        axes[i].set_xlabel('Observed')
        axes[i].set_ylabel('Transformed')
        axes[i].set_title(col)

    # response transformation
    y_obs_arr = np.asarray(y_obs)
    order = np.argsort(y_obs_arr)
    axes[len(cols)].plot(y_obs_arr[order], yhat[order], 'k-')
    axes[len(cols)].set_xlabel('Observed')
    axes[len(cols)].set_ylabel('Transformed')

    y_name = 'Response'
    if hasattr(y_obs, 'name') and y_obs.name:
        y_name = str(y_obs.name)
    axes[len(cols)].set_title(y_name)

    # hide unused panels
    for j in range(nplots, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('MORALS', fontweight='bold')
    plt.tight_layout()
    return fig


def plot_object_scores(model, dim=None, ax=None, **kwargs):
    """
    Generic 2D scatter plot of object scores for any fitted Gifi model.

    Parameters
    ----------
    model : fitted Gifi estimator (Princals, Homals, or Morals)
    dim : list of 2 ints, default=[0, 1]
    ax : matplotlib.axes.Axes, optional
        Axes to plot into. Creates new figure if None.
    **kwargs : additional keyword arguments passed to ax.scatter()
    """
    if dim is None:
        dim = [0, 1]
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    scores = model.object_scores_
    if scores is None:
        raise ValueError("Model does not have 'object_scores_'. Ensure it is fitted.")

    # Only plot if we have enough dimensions
    if scores.shape[1] <= dim[1]:
        raise ValueError(f"Dim {dim} out of bounds for {scores.shape[1]}D object scores.")

    kwargs.setdefault('color', 'steelblue')
    kwargs.setdefault('alpha', 0.6)
    kwargs.setdefault('s', 20)
    kwargs.setdefault('zorder', 3)

    ax.scatter(scores[:, dim[0]], scores[:, dim[1]], label='Objects', **kwargs)

    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel(f'Dimension {dim[0] + 1}')
    ax.set_ylabel(f'Dimension {dim[1] + 1}')
    ax.set_title(f'{model.__class__.__name__}: Object Scores')

    return ax


def plot_quantifications(model, dim=None, ax=None, **kwargs):
    """
    Generic 2D scatter plot of category quantifications.

    Parameters
    ----------
    model : fitted Gifi estimator (Princals, Homals)
    dim : list of 2 ints, default=[0, 1]
    ax : matplotlib.axes.Axes, optional
    **kwargs : additional arguments
    """
    if dim is None:
        dim = [0, 1]
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    quants = model.category_quantifications_
    if quants is None:
        raise ValueError(
            "Model does not have 'category_quantifications_'. (Note: Morals doesn't compute standard quantifications).")

    # Handle dictionary (Homals) or list (Princals)
    if isinstance(quants, dict):
        q_list = list(quants.values())
        names = list(quants.keys())
    else:
        q_list = quants
        names = [f"Var{j}" for j in range(len(q_list))]

    var_colors = plt.cm.Set1(np.linspace(0, 1, len(q_list)))

    # Try to grab original variables to get category labels
    data = None
    if hasattr(model, 'result_') and 'data' in model.result_:
        data = model.result_['data']

    for j, (name, q) in enumerate(zip(names, q_list)):
        q = np.asarray(q)
        if q.ndim == 1:
            q = q[:, None]

        if q.shape[1] <= dim[1]:
            padded = np.zeros((q.shape[0], dim[1] + 1))
            padded[:, :q.shape[1]] = q
            q = padded

        n_cats = q.shape[0]
        cat_labels = list(range(n_cats))

        if data is not None and j < len(data.columns):
            col = data.iloc[:, j]
            if hasattr(col, 'cat'):
                cat_labels = col.cat.categories.tolist()
            else:
                cat_labels = sorted(col.unique())

        for i in range(min(n_cats, len(cat_labels))):
            ax.scatter(q[i, dim[0]], q[i, dim[1]],
                       color=var_colors[j], marker='^', s=60,
                       edgecolors='black', linewidths=0.5, zorder=5)
            ax.annotate(str(cat_labels[i]),
                        xy=(q[i, dim[0]], q[i, dim[1]]),
                        xytext=(4, 4), textcoords='offset points',
                        fontsize=7, color=var_colors[j], fontweight='bold')

    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel(f'Dimension {dim[0] + 1}')
    ax.set_ylabel(f'Dimension {dim[1] + 1}')
    ax.set_title(f'{model.__class__.__name__}: Category Quantifications')

    return ax


def plot_biplot(model, dim=None, ax=None, **kwargs):
    """
    Generic 2D biplot of object scores and component loadings.

    Parameters
    ----------
    model : fitted Gifi estimator (Princals, Homals)
    dim : list of 2 ints, default=[0, 1]
    ax : matplotlib.axes.Axes, optional
    """
    if dim is None:
        dim = [0, 1]
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    # Plot scores first
    plot_object_scores(model, dim=dim, ax=ax, **kwargs)

    loadings = model.component_loadings_
    scores = model.object_scores_

    if loadings is None or scores is None:
        return ax

    if isinstance(loadings, dict):
        loadings_arr = np.vstack(list(loadings.values()))  # stack dict values (e.g., from Homals)
    else:
        loadings_arr = np.asarray(loadings)

    # Check if we have 2 dimensions to plot arrows
    if loadings_arr.ndim > 1 and loadings_arr.shape[1] > dim[1]:
        # Scale arrows to fit nicely within objects
        scalef = np.percentile(np.abs(scores), 80) / (np.max(np.abs(loadings_arr)) + 1e-12)

        for i in range(loadings_arr.shape[0]):
            ax.annotate('',
                        xy=(loadings_arr[i, dim[0]] * scalef,
                            loadings_arr[i, dim[1]] * scalef),
                        xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color='coral', lw=1.5))

    ax.set_title(f'{model.__class__.__name__}: Biplot')
    return ax


# =====================================================================
# Unified R-style plotting dispatcher and helpers
# =====================================================================

def _get_result(result_or_model):
    """Extract result dictionary from model or return dict."""
    if isinstance(result_or_model, dict):
        return result_or_model
    return getattr(result_or_model, 'result_', result_or_model)


def _plot_screeplot(result, ax=None, title=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
    else:
        fig = ax.figure
    
    evals = np.asarray(result.get('evals', []))
    if len(evals) == 0:
        raise ValueError("No eigenvalues found in result.")
        
    ax.bar(np.arange(len(evals)) + 1, evals, color='steelblue', alpha=0.7)
    ax.set_xlabel('Dimension')
    ax.set_ylabel('Eigenvalue')
    ax.set_title(title if title else 'Scree Plot')
    ax.set_xticks(np.arange(len(evals)) + 1)
    return fig


def _plot_loadplot(result, dim=(0, 1), ax=None, title=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
        
    loadings = result.get('loadings')
    if loadings is None:
        raise ValueError("No loadings found in result.")
        
    # extract (n_vars, 2) array of loadings
    if isinstance(loadings, dict):
        # Homals list of (ndim, copies) - take first column of each
        names = list(loadings.keys())
        loadings_arr = np.vstack([loadings[k][:, 0] for k in names])
    else:
        # Princals (n_vars, ndim)
        loadings_arr = np.asarray(loadings)
        data = result.get('data')
        names = list(data.columns) if data is not None and hasattr(data, 'columns') else [f"Var{i}" for i in range(loadings_arr.shape[0])]

    if loadings_arr.ndim < 2 or loadings_arr.shape[1] <= dim[1]:
        raise ValueError(f"Not enough dimensions to plot {dim}")

    max_val = np.max(np.abs(loadings_arr[:, [dim[0], dim[1]]]))
    if max_val == 0:  # Avoid singular limits
        max_val = 1
    ax.set_xlim(-max_val * 1.2, max_val * 1.2)
    ax.set_ylim(-max_val * 1.2, max_val * 1.2)
    
    for i, name in enumerate(names):
        x, y = loadings_arr[i, dim[0]], loadings_arr[i, dim[1]]
        ax.annotate(name, xy=(x, y), xytext=(x * 1.05, y * 1.05),
                    fontsize=9, color='coral', ha='center', va='center')
        ax.annotate('', xy=(x, y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='coral', lw=1.5))
                    
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel(f'Dimension {dim[0] + 1}')
    ax.set_ylabel(f'Dimension {dim[1] + 1}')
    ax.set_title(title if title else 'Loadings')
    return ax


def _plot_vecplot(result, dim=(0, 1), ax=None, title=None, **kwargs):
    """Same as loadplot, but typically called for Homals."""
    return _plot_loadplot(result, dim=dim, ax=ax, title=(title if title else 'Variable Vectors'), **kwargs)


def _plot_biplot(result, dim=(0, 1), ax=None, title=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    scores = np.asarray(result.get('objectscores', []))
    if len(scores) == 0:
        raise ValueError("No object scores in result.")
        
    ax.scatter(scores[:, dim[0]], scores[:, dim[1]],
               color='steelblue', alpha=0.6, s=20, zorder=3, label='Objects')

    loadings = result.get('loadings')
    if loadings is not None:
        if isinstance(loadings, dict):
             loadings_arr = np.vstack([loadings[k][:, 0] for k in loadings.keys()])
        else:
             loadings_arr = np.asarray(loadings)
             
        if loadings_arr.ndim > 1 and loadings_arr.shape[1] > dim[1]:
            scalef = np.percentile(np.abs(scores), 80) / (np.max(np.abs(loadings_arr)) + 1e-12)
            for i in range(loadings_arr.shape[0]):
                ax.annotate('',
                            xy=(loadings_arr[i, dim[0]] * scalef, loadings_arr[i, dim[1]] * scalef),
                            xytext=(0, 0),
                            arrowprops=dict(arrowstyle='->', color='coral', lw=1.5))

    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel(f'Dimension {dim[0] + 1}')
    ax.set_ylabel(f'Dimension {dim[1] + 1}')
    ax.set_title(title if title else 'Biplot')
    return ax


def _plot_transplot(result, title=None, ncols=2, **kwargs):
    transforms = result.get('transform')
    data = result.get('data')
    datanum = result.get('datanum')
    
    if transforms is None or datanum is None:
        raise ValueError("Transformation data not found in result.")

    # Normalize transforms format
    if isinstance(transforms, list):
        # Homals list
        tr_list = [np.asarray(tr)[:, 0] for tr in transforms]
    else:
        # Princals matrix (nobs, nvars)
        if transforms.ndim == 1:
            tr_list = [transforms]
        else:
            tr_list = [transforms[:, i] for i in range(transforms.shape[1])]

    names = list(data.columns) if data is not None and hasattr(data, 'columns') else [f"Var{i}" for i in range(len(tr_list))]
    nplots = len(tr_list)
    nrows = int(np.ceil(nplots / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).flatten()
    var_colors = plt.cm.Set1(np.linspace(0, 1, nplots))
    
    for j, tr in enumerate(tr_list):
        col_data = np.asarray(datanum[:, j])
        sorted_idx = np.argsort(col_data)
        axes[j].step(
            col_data[sorted_idx],
            tr[sorted_idx],
            where='mid',
            color=var_colors[j],
            alpha=0.8)
        axes[j].set_xlabel('Original Values (quantified)')
        axes[j].set_ylabel('Transformed Scale')
        axes[j].set_title(names[j])
        
    for j in range(nplots, len(axes)):
        axes[j].set_visible(False)
        
    fig.suptitle(title if title else 'Transformation Plot', fontweight='bold')
    plt.tight_layout()
    return fig


def _plot_objplot(result, dim=(0, 1), ax=None, title=None, group=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    scores = np.asarray(result.get('objectscores', []))
    if len(scores) == 0:
        raise ValueError("No object scores in result.")
        
    if group is not None:
        group_arr = np.asarray(group)
        if len(group_arr) != scores.shape[0]:
            raise ValueError(f"group length ({len(group_arr)}) != object scores length ({scores.shape[0]})")
        
        unique_groups = np.unique(group_arr)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_groups)))
        for i, grp in enumerate(unique_groups):
            idx = (group_arr == grp)
            ax.scatter(scores[idx, dim[0]], scores[idx, dim[1]],
                       color=colors[i], label=str(grp), alpha=0.6, s=20)
        ax.legend(title='Group')
    else:
        ax.scatter(scores[:, dim[0]], scores[:, dim[1]],
                   color='steelblue', alpha=0.6, s=20, label='Objects')

    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel(f'Dimension {dim[0] + 1}')
    ax.set_ylabel(f'Dimension {dim[1] + 1}')
    ax.set_title(title if title else 'Object Scores')
    return ax


def _plot_prjplot(result, dim=(0, 1), ax=None, title=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
        
    cat_centroids = result.get('cat.centroids')
    if cat_centroids is None:
        raise ValueError("cat.centroids not found. prjplot is typically for Homals only.")
        
    data = result.get('data')
    var_colors = plt.cm.Set1(np.linspace(0, 1, len(cat_centroids)))
    
    for j, centroids in enumerate(cat_centroids):
        c_arr = np.asarray(centroids)
        if c_arr.ndim < 2 or c_arr.shape[1] <= dim[1]:
            continue
            
        col = data.iloc[:, j] if data is not None else None
        if col is not None and hasattr(col, 'cat'):
            labels = col.cat.categories.tolist()
        elif col is not None:
             labels = sorted(col.unique())
        else:
             labels = list(range(c_arr.shape[0]))
             
        for i in range(min(c_arr.shape[0], len(labels))):
            ax.scatter(c_arr[i, dim[0]], c_arr[i, dim[1]],
                       color=var_colors[j], marker='^', s=60,
                       edgecolors='black', linewidths=0.5, zorder=5)
            ax.annotate(str(labels[i]),
                        xy=(c_arr[i, dim[0]], c_arr[i, dim[1]]),
                        xytext=(4, 4), textcoords='offset points',
                        fontsize=7, color=var_colors[j], fontweight='bold')
                        
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel(f'Dimension {dim[0] + 1}')
    ax.set_ylabel(f'Dimension {dim[1] + 1}')
    ax.set_title(title if title else 'Projected Category Points')
    return ax


def plot(result, plot_type='loadplot', plot_dim=(0, 1), group=None, title=None, ax=None, **kwargs):
    """
    Unified plotting dispatcher for pyGifi models (Princals, Homals).
    
    Mirrors R's S3 plot.homals and plot.princals functionality.
    
    Parameters
    ----------
    result : model instance or dict
        Fitted model (e.g., Homals(), Princals()) or its `.result_` dictionary.
    plot_type : str, default='loadplot'
        The type of plot to generate. Options:
        - 'loadplot' : Variable loadings vectors (Princals/Homals)
        - 'biplot'   : Object scores + scaled loading vectors (Princals/Homals)
        - 'transplot': Subplots of original vs. transformed values (Both)
        - 'objplot'  : Object scores optionally grouped by a variable (Both)
        - 'prjplot'  : Category centroids in object score space (Homals)
        - 'vecplot'  : Variable vectors in object score space (Homals)
        - 'screeplot': Eigenvalue bar chart (Both)
    plot_dim : tuple of 2 ints, default=(0, 1)
        Dimensions to plot on x and y axes (0-indexed).
    group : array-like, optional
        Grouping variable used to color points in 'objplot'.
    title : str, optional
        Custom title for the plot.
    ax : matplotlib.axes.Axes, optional
        Axes to plot into (for plot types that generate a single axes).
        For 'transplot' and 'screeplot', a new Figure is typically generated.
    **kwargs : additional keyword arguments
        Passed down to the specific plot helper.
        
    Returns
    -------
    matplotlib.axes.Axes or matplotlib.figure.Figure
    """
    res = _get_result(result)
    
    dispatch_map = {
        'loadplot': _plot_loadplot,
        'biplot': _plot_biplot,
        'transplot': _plot_transplot,
        'objplot': _plot_objplot,
        'screeplot': _plot_screeplot,
        'prjplot': _plot_prjplot,
        'vecplot': _plot_vecplot
    }
    
    if plot_type not in dispatch_map:
        valid = ", ".join(f"'{k}'" for k in dispatch_map.keys())
        raise ValueError(f"Unknown plot_type '{plot_type}'. Valid options are: {valid}")
        
    func = dispatch_map[plot_type]
    
    return func(res, dim=plot_dim, group=group, title=title, ax=ax, **kwargs)
