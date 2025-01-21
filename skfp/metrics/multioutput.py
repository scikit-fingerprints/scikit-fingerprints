from importlib.metadata import version
from typing import Callable, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.utils._param_validation import validate_params

from skfp.metrics.auroc import auroc_score
from skfp.metrics.spearman import spearman_correlation


@validate_params(
    {
        "y_true": ["array-like"],
        "y_pred": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def multioutput_accuracy_score(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    *args,
    **kwargs,
) -> float:
    """
    Accuracy score for multioutput problems.

    Returns the average value over all tasks. Missing values in target labels are
    ignored. Also supports single-task evaluation.

    Any additional arguments are passed to the underlying ``accuracy_score`` function,
    see `scikit-learn documentation <sklearn>`_ for more information.

    .. _sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    *args, **kwargs
        Any additional parameters for the underlying scikit-learn metric function.

    Returns
    -------
    score : float or int
        Average accuracy value over all tasks.

    Examples
    --------
    >>> import numpy as np
    >>> from skfp.metrics import multioutput_accuracy_score
    >>> y_true = [[0, 0], [1, 1]]
    >>> y_pred = [[0, 0], [0, 1]]
    >>> multioutput_accuracy_score(y_true, y_pred)
    0.75
    >>> y_true = [[0, np.nan], [1, np.nan], [np.nan, np.nan]]
    >>> y_pred = [[0, 0], [0, 0], [1, 0]]
    >>> multioutput_accuracy_score(y_true, y_pred)
    0.5
    """
    return _safe_multioutput_metric(accuracy_score, y_true, y_pred, *args, **kwargs)


@validate_params(
    {
        "y_true": ["array-like"],
        "y_score": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def multioutput_auroc_score(
    y_true: Union[np.ndarray, list],
    y_score: Union[np.ndarray, list],
    *args,
    **kwargs,
) -> float:
    """
    Area Under Receiver Operating Characteristic curve (AUROC / ROC AUC) score for
    multioutput problems.

    Returns the average value over all tasks. Missing values in target labels are
    ignored. Columns with constant true value are ignored by default, but can also
    use default value - see ``auroc_score`` function. As such, it can be safely used
    e.g. in cross-validation. Also supports single-task evaluation.

    For scikit-learn <1.6, ``auroc_score`` function from scikit-fingerprints is used,
    and for newer versions >=1.6 ``roc_auc_score`` from scikit-learn is used instead.
    See `scikit-learn documentation <sklearn>`_ for more information.

    .. _sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_score : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Target scores, i.e. probability of the class with the greater label for each
        output** of the classifier.

    *args, **kwargs
        Any additional parameters for the underlying scikit-learn metric function.

    Returns
    -------
    score : float
        Average AUROC value over all tasks.

    Examples
    --------
    >>> import numpy as np
    >>> from skfp.metrics import multioutput_auroc_score
    >>> y_true = np.array([[0, 0], [1, 1]])
    >>> y_score = np.array([[0.75, 0.0], [0.9, 0.0]])
    >>> multioutput_auroc_score(y_true, y_score)
    0.75
    >>> y_true = np.array([[0, 0], [1, np.nan], [np.nan, 1]])
    >>> y_score = np.array([[0.75, 0.0], [0.25, 0.0], [0.0, 0.25]])
    >>> multioutput_auroc_score(y_true, y_score)
    0.5
    """
    sklearn_ver = version("scikit-learn")  # e.g. 1.6.0
    sklearn_ver = ".".join(sklearn_ver.split(".")[:-1])  # e.g. 1.6
    if float(sklearn_ver) < 1.6:
        func = auroc_score
    else:
        func = roc_auc_score

    return _safe_multioutput_metric(func, y_true, y_score, *args, **kwargs)


@validate_params(
    {
        "y_true": ["array-like"],
        "y_score": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def multioutput_auprc_score(
    y_true: Union[np.ndarray, list],
    y_score: Union[np.ndarray, list],
    *args,
    **kwargs,
) -> float:
    """
    Area Under Precision-Recall Curve (AUPRC / AUC PRC / average precision) score for
    multioutput problems.

    Returns the average value over all tasks. Missing values in target labels are
    ignored. Also supports single-task evaluation.

    Any additional arguments are passed to the underlying ``average_precision_score``
    function, see `scikit-learn documentation <sklearn>`_ for more information.

    .. _sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_score : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Target scores, i.e. probability of the class with the greater label for each
        output** of the classifier.

    *args, **kwargs
        Any additional parameters for the underlying scikit-learn metric function.

    Returns
    -------
    score : float
        Average AUPRC value over all tasks.

    Examples
    --------
    >>> import numpy as np
    >>> from skfp.metrics import multioutput_auprc_score
    >>> y_true = [[0, 0], [1, 1]]
    >>> y_score = [[0.75, 0.0], [0.9, 0.0]]
    >>> multioutput_auprc_score(y_true, y_score)
    0.75
    >>> y_true = [[0, 0], [1, np.nan], [np.nan, 1]]
    >>> y_score = [[0.75, 0.0], [0.25, 0.0], [0.0, 0.25]]
    >>> multioutput_auprc_score(y_true, y_score)
    0.75
    """
    return _safe_multioutput_metric(
        average_precision_score, y_true, y_score, *args, **kwargs
    )


@validate_params(
    {
        "y_true": ["array-like"],
        "y_pred": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def multioutput_balanced_accuracy_score(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    *args,
    **kwargs,
) -> float:
    """
    Balanced accuracy (average recall) score for multioutput problems.

    Returns the average value over all tasks. Missing values in target labels are
    ignored. Also supports single-task evaluation.

    Any additional arguments are passed to the underlying ``balanced_accuracy_score``
    function, see `scikit-learn documentation <sklearn>`_ for more information.

    .. _sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    *args, **kwargs
        Any additional parameters for the underlying scikit-learn metric function.

    Returns
    -------
    score : float
        Average balanced accuracy value over all tasks.

    Examples
    --------
    >>> import numpy as np
    >>> from skfp.metrics import multioutput_balanced_accuracy_score
    >>> y_true = [[0, 0], [1, 1]]
    >>> y_pred = [[0, 0], [0, 1]]
    >>> multioutput_balanced_accuracy_score(y_true, y_pred)
    0.75
    >>> y_true = [[0, np.nan], [1, np.nan], [np.nan, np.nan]]
    >>> y_pred = [[0, 0], [0, 0], [1, 0]]
    >>> multioutput_balanced_accuracy_score(y_true, y_pred)
    0.5
    """
    return _safe_multioutput_metric(
        balanced_accuracy_score, y_true, y_pred, *args, **kwargs
    )


@validate_params(
    {
        "y_true": ["array-like"],
        "y_pred": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def multioutput_cohen_kappa_score(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    *args,
    **kwargs,
) -> float:
    """
    Cohen's kappa score for multioutput problems.

    Returns the average value over all tasks. Missing values in target labels are
    ignored. Also supports single-task evaluation.

    Any additional arguments are passed to the underlying ``cohen_kappa_score``
    function, see `scikit-learn documentation <sklearn>`_ for more information.

    .. _sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    *args, **kwargs
        Any additional parameters for the underlying scikit-learn metric function.

    Returns
    -------
    score : float
        Average Cohen's kappa value over all tasks.

    Examples
    --------
    >>> import numpy as np
    >>> from skfp.metrics import multioutput_cohen_kappa_score
    >>> y_true = [[0, 0], [1, 1]]
    >>> y_pred = [[0, 0], [0, 1]]
    >>> multioutput_cohen_kappa_score(y_true, y_pred)
    0.5
    >>> y_true = [[0, np.nan], [1, np.nan], [np.nan, np.nan]]
    >>> y_pred = [[0, 0], [0, 0], [1, 0]]
    >>> multioutput_cohen_kappa_score(y_true, y_pred)
    0.0
    """
    return _safe_multioutput_metric(cohen_kappa_score, y_true, y_pred, *args, **kwargs)


@validate_params(
    {
        "y_true": ["array-like"],
        "y_pred": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def multioutput_f1_score(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    *args,
    **kwargs,
) -> float:
    """
    F1 score for multioutput problems.

    Returns the average value over all tasks. Missing values in target labels are
    ignored. Columns with constant true value are also ignored, which differs from
    default scikit-learn behavior (it returns value 0 by default). Also supports
    single-task evaluation.

    Any additional arguments are passed to the underlying ``f1_score`` function,
    see `scikit-learn documentation <sklearn>`_ for more information.

    .. _sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    *args, **kwargs
        Any additional parameters for the underlying scikit-learn metric function.

    Returns
    -------
    score : float
        Average F1 score value over all tasks.

    Examples
    --------
    >>> import numpy as np
    >>> from skfp.metrics import multioutput_f1_score
    >>> y_true = [[0, 0], [1, 1]]
    >>> y_pred = [[0, 0], [0, 1]]
    >>> multioutput_f1_score(y_true, y_pred)
    0.5
    >>> y_true = [[0, np.nan], [1, np.nan], [np.nan, np.nan]]
    >>> y_pred = [[0, 0], [0, 0], [1, 0]]
    >>> multioutput_f1_score(y_true, y_pred)
    0.0
    """
    return _safe_multioutput_metric(f1_score, y_true, y_pred, *args, **kwargs)


@validate_params(
    {
        "y_true": ["array-like"],
        "y_pred": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def multioutput_matthews_corr_coef(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    *args,
    **kwargs,
) -> float:
    """
    Matthews Correlation Coefficient (MCC) for multioutput problems.

    Returns the average value over all tasks. Missing values in target labels are
    ignored. Also supports single-task evaluation.

    Any additional arguments are passed to the underlying ``matthews_corrcoef``
    function, see `scikit-learn documentation <sklearn>`_ for more information.

    .. _sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    *args, **kwargs
        Any additional parameters for the underlying scikit-learn metric function.

    Returns
    -------
    score : float
        Average MCC value over all tasks.

    Examples
    --------
    >>> import numpy as np
    >>> from skfp.metrics import multioutput_matthews_corr_coef
    >>> y_true = [[0, 0], [1, 1]]
    >>> y_pred = [[0, 0], [0, 1]]
    >>> multioutput_matthews_corr_coef(y_true, y_pred)
    0.5
    >>> y_true = [[0, np.nan], [1, np.nan], [np.nan, np.nan]]
    >>> y_pred = [[0, 0], [0, 0], [1, 0]]
    >>> multioutput_matthews_corr_coef(y_true, y_pred)
    0.0
    """
    return _safe_multioutput_metric(matthews_corrcoef, y_true, y_pred, *args, **kwargs)


@validate_params(
    {
        "y_true": ["array-like"],
        "y_pred": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def multioutput_mean_absolute_error(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    *args,
    **kwargs,
) -> float:
    """
    Mean absolute error (MAE) for multioutput problems.

    Returns the average value over all tasks. Missing values in target labels are
    ignored. Also supports single-task evaluation.

    Any additional arguments are passed to the underlying ``mean_absolute_error``
    function, see `scikit-learn documentation <sklearn>`_ for more information.

    .. _sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    *args, **kwargs
        Any additional parameters for the underlying scikit-learn metric function.

    Returns
    -------
    score : float
        Average MAE value over all tasks.

    Examples
    --------
    >>> import numpy as np
    >>> from skfp.metrics import multioutput_mean_absolute_error
    >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
    >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
    >>> multioutput_mean_absolute_error(y_true, y_pred)
    0.75
    >>> y_true = [[0.5, 1], [-1, 1], [np.nan, 10], [-10, np.nan]]
    >>> y_pred = [[0, 2], [-1, 2], [-3, 8], [-10, 5]]
    >>> multioutput_mean_absolute_error(y_true, y_pred)
    0.75
    """
    return _safe_multioutput_metric(
        mean_absolute_error, y_true, y_pred, *args, **kwargs
    )


@validate_params(
    {
        "y_true": ["array-like"],
        "y_pred": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def multioutput_mean_squared_error(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    *args,
    **kwargs,
) -> float:
    """
    Mean squared error (MSE) for multioutput problems.

    Returns the average value over all tasks. Missing values in target labels are
    ignored. Also supports single-task evaluation.

    Any additional arguments are passed to the underlying ``mean_squared_error``
    function, see `scikit-learn documentation <sklearn>`_ for more information.

    .. _sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    *args, **kwargs
        Any additional parameters for the underlying scikit-learn metric function.

    Returns
    -------
    score : float
        Average MSE value over all tasks.

    Examples
    --------
    >>> import numpy as np
    >>> from skfp.metrics import multioutput_mean_squared_error
    >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
    >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
    >>> multioutput_mean_squared_error(y_true, y_pred)
    0.708...
    >>> y_true = [[0.5, 1], [-1, 1], [np.nan, 10], [-10, np.nan]]
    >>> y_pred = [[0, 2], [-1, 2], [-3, 8], [-10, 5]]
    >>> multioutput_mean_squared_error(y_true, y_pred)
    1.041...
    """
    return _safe_multioutput_metric(mean_squared_error, y_true, y_pred, *args, **kwargs)


@validate_params(
    {
        "y_true": ["array-like"],
        "y_pred": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def multioutput_precision_score(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    *args,
    **kwargs,
) -> float:
    """
    Precision score for multioutput problems.

    Returns the average value over all tasks. Missing values in target labels are
    ignored. Warnings are not raised for columns with constant false prediction,
    and 0.0 is assumed. Also supports single-task evaluation.

    Any additional arguments are passed to the underlying ``precision_score``
    function, see `scikit-learn documentation <sklearn>`_ for more information.

    .. _sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    *args, **kwargs
        Any additional parameters for the underlying scikit-learn metric function.

    Returns
    -------
    score : float
        Average precision value over all tasks.

    Examples
    --------
    >>> import numpy as np
    >>> from skfp.metrics import multioutput_precision_score
    >>> y_true = [[0, 0], [1, 1]]
    >>> y_pred = [[0, 0], [0, 1]]
    >>> multioutput_precision_score(y_true, y_pred)
    0.5
    >>> y_true = [[0, np.nan], [1, np.nan], [np.nan, np.nan]]
    >>> y_pred = [[0, 0], [0, 0], [1, 0]]
    >>> multioutput_precision_score(y_true, y_pred)
    0.0
    """
    return _safe_multioutput_metric(
        precision_score,
        y_true,
        y_pred,
        *args,
        zero_division=0.0,
        **kwargs,
    )


@validate_params(
    {
        "y_true": ["array-like"],
        "y_pred": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def multioutput_recall_score(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    *args,
    **kwargs,
) -> float:
    """
    Recall score for multioutput problems.

    Returns the average value over all tasks. Missing values in target labels are
    ignored. Also supports single-task evaluation.

    Any additional arguments are passed to the underlying ``recall_score`` function,
    see `scikit-learn documentation <sklearn>`_ for more information.

    .. _sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    *args, **kwargs
        Any additional parameters for the underlying scikit-learn metric function.

    Returns
    -------
    score : float
        Average recall value over all tasks.

    Examples
    --------
    >>> import numpy as np
    >>> from skfp.metrics import multioutput_recall_score
    >>> y_true = [[0, 0], [1, 1]]
    >>> y_pred = [[0, 0], [0, 1]]
    >>> multioutput_recall_score(y_true, y_pred)
    0.5
    >>> y_true = [[0, np.nan], [1, np.nan], [np.nan, np.nan]]
    >>> y_pred = [[0, 0], [0, 0], [1, 0]]
    >>> multioutput_recall_score(y_true, y_pred)
    0.0
    """
    return _safe_multioutput_metric(recall_score, y_true, y_pred, *args, **kwargs)


@validate_params(
    {
        "y_true": ["array-like"],
        "y_pred": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def multioutput_root_mean_squared_error(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    *args,
    **kwargs,
) -> float:
    """
    Root mean squared error (RMSE) for multioutput problems.

    Returns the average value over all tasks. Missing values in target labels are
    ignored. Also supports single-task evaluation.

    Any additional arguments are passed to the underlying ``root_mean_squared_error``
    function, see `scikit-learn documentation <sklearn>`_ for more information.

    .. _sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.root_mean_squared_error.html

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    *args, **kwargs
        Any additional parameters for the underlying scikit-learn metric function.

    Returns
    -------
    score : float
        Average RMSE value over all tasks.

    Examples
    --------
    >>> import numpy as np
    >>> from skfp.metrics import multioutput_root_mean_squared_error
    >>> y_true = [[0.5, 1], [-1, 1], [7, -6]]
    >>> y_pred = [[0, 2], [-1, 2], [8, -5]]
    >>> multioutput_root_mean_squared_error(y_true, y_pred)
    0.822...
    >>> y_true = [[0.5, 1], [-1, 1], [np.nan, 10], [-10, np.nan]]
    >>> y_pred = [[0, 2], [-1, 2], [-3, 8], [-10, 5]]
    >>> multioutput_root_mean_squared_error(y_true, y_pred)
    0.851...
    """
    return _safe_multioutput_metric(
        root_mean_squared_error, y_true, y_pred, *args, **kwargs
    )


@validate_params(
    {
        "y_true": ["array-like"],
        "y_pred": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def multioutput_spearman_correlation(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    *args,
    **kwargs,
) -> float:
    """
    Spearman correlation for multioutput problems.

    Returns the average value over all tasks. Missing values in target labels are
    ignored. Also supports single-task evaluation.

    Any additional arguments are passed to the underlying ``spearman_correlation``
    function, see :func:`spearman_correlation` for more information.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    *args, **kwargs
        Any additional parameters for the underlying scikit-learn metric function.

    Returns
    -------
    score : float
        Average Spearman correlation value over all tasks.

    Examples
    --------
    >>> import numpy as np
    >>> from skfp.metrics import multioutput_spearman_correlation
    >>> y_true = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    >>> y_pred = np.array([[1, 4], [2, 3], [3, 2], [4, 1]])
    >>> multioutput_spearman_correlation(y_true, y_pred)
    0.0
    >>> y_true = np.array([[1, 1], [np.nan, 2], [2, np.nan], [3, 3]])
    >>> y_pred = np.array([[1, 1], [0, 3], [3, 0], [4, 2]])
    >>> multioutput_spearman_correlation(y_true, y_pred)
    0.75
    """
    return _safe_multioutput_metric(
        spearman_correlation, y_true, y_pred, *args, **kwargs
    )


def _safe_multioutput_metric(
    metric: Callable,
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    *args,
    **kwargs,
) -> float:
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)

    # make sure both arrays are 2D NumPy arrays with the same shape
    y_true, y_pred = _ensure_proper_shapes(y_true, y_pred)

    values = []
    for i in range(y_true.shape[1]):
        y_true_i = y_true[:, i]
        y_pred_i = y_pred[:, i]

        # in case of all-NaN column, skip it
        if np.all(np.isnan(y_true_i)):
            continue

        # remove NaN values
        non_nan_mask = ~np.isnan(y_true_i) & ~np.isnan(y_pred_i)
        y_true_i = y_true_i[non_nan_mask]
        y_pred_i = y_pred_i[non_nan_mask]

        col_value = metric(y_true_i, y_pred_i, *args, **kwargs)
        values.append(col_value)

    if not values:
        raise ValueError(
            "Could not compute metric value, y_true had only "
            "missing or constant values in all columns."
        )

    return float(np.mean(values))


def _ensure_proper_shapes(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Both true labels and predictions must have the same shape, got: "
            f"true labels {y_true.ndim}, predictions {y_pred.shape}"
        )

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    elif y_true.ndim > 2:
        raise ValueError(
            f"True labels must have 1 or 2 dimensions, got shape {y_true.shape}"
        )

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    elif y_pred.ndim > 2:
        raise ValueError(
            f"Predictions must have 1 or 2 dimensions, got shape {y_true.shape}"
        )

    return y_true, y_pred
