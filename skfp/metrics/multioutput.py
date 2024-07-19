from typing import Callable

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


def multioutput_accuracy_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *args,
    **kwargs,
) -> float:
    return _safe_multioutput_metric(accuracy_score, y_true, y_pred, *args, **kwargs)


def multioutput_auroc_score(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *args,
    **kwargs,
) -> float:
    return _safe_multioutput_metric(
        roc_auc_score,
        y_true,
        y_score,
        True,
        *args,
        **kwargs,
    )


def multioutput_auprc_score(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *args,
    **kwargs,
) -> float:
    # scikit-learn calls AUPRC "average precision"
    return _safe_multioutput_metric(
        average_precision_score, y_true, y_score, *args, **kwargs
    )


def multioutput_balanced_accuracy_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *args,
    **kwargs,
) -> float:
    return _safe_multioutput_metric(
        balanced_accuracy_score, y_true, y_pred, *args, **kwargs
    )


def multioutput_cohen_kappa_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *args,
    **kwargs,
) -> float:
    return _safe_multioutput_metric(cohen_kappa_score, y_true, y_pred, *args, **kwargs)


def multioutput_f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *args,
    **kwargs,
) -> float:
    return _safe_multioutput_metric(f1_score, y_true, y_pred, *args, **kwargs)


def multioutput_matthews_corr_coef(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *args,
    **kwargs,
) -> float:
    return _safe_multioutput_metric(matthews_corrcoef, y_true, y_pred, *args, **kwargs)


def multioutput_mean_absolute_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *args,
    **kwargs,
) -> float:
    return _safe_multioutput_metric(
        mean_absolute_error, y_true, y_pred, *args, **kwargs
    )


def multioutput_mean_squared_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *args,
    **kwargs,
) -> float:
    return _safe_multioutput_metric(mean_squared_error, y_true, y_pred, *args, **kwargs)


def multioutput_precision_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *args,
    **kwargs,
) -> float:
    return _safe_multioutput_metric(precision_score, y_true, y_pred, *args, **kwargs)


def multioutput_recall_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *args,
    **kwargs,
) -> float:
    return _safe_multioutput_metric(recall_score, y_true, y_pred, *args, **kwargs)


def multioutput_root_mean_squared_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *args,
    **kwargs,
) -> float:
    return _safe_multioutput_metric(
        root_mean_squared_error, y_true, y_pred, *args, **kwargs
    )


def _safe_multioutput_metric(
    metric: Callable,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    omit_constant_cols: bool = False,
    *args,
    **kwargs,
) -> float:
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    elif y_true.ndim > 2:
        raise ValueError(f"True labels must have 1 or 2 dimensions, got {y_true.ndim}")
    elif y_true.ndim == 0:
        raise ValueError(f"Expected matrix for true labels, got a scalar {y_true}")

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    elif y_pred.ndim > 2:
        raise ValueError(f"Predictions must have 1 or 2 dimensions, got {y_pred.ndim}")
    elif y_pred.ndim == 0:
        raise ValueError(f"Expected matrix for predictions, got a scalar {y_pred}")

    values = []
    for i in range(y_true.shape[1]):
        y_true_i = y_true[:, i]
        y_pred_i = y_pred[:, i]

        # in case of all-NaN column, skip it
        if np.all(np.isnan(y_true_i)):
            continue

        # omit constant columns for metrics not supporting those, e.g. AUROC
        if len(np.unique(y_true_i)) == 1 and omit_constant_cols:
            continue

        # remove NaN values
        non_nan_mask = ~np.isnan(y_true_i) & ~np.isnan(y_pred_i)
        y_true_i = y_true_i[non_nan_mask]
        y_pred_i = y_pred_i[non_nan_mask]

        col_value = metric(y_true_i, y_pred_i, *args, **kwargs)
        values.append(col_value)

    if not values:
        raise ValueError(
            "Could not compute metric value, y_true had only missing or constant values."
        )

    return np.mean(values)
