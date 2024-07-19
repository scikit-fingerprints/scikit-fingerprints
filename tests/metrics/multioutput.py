import warnings
from typing import Callable

import numpy as np
import pytest
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

from skfp.metrics import (
    multioutput_accuracy_score,
    multioutput_auprc_score,
    multioutput_auroc_score,
    multioutput_balanced_accuracy_score,
    multioutput_cohen_kappa_score,
    multioutput_f1_score,
    multioutput_matthews_corr_coef,
    multioutput_mean_absolute_error,
    multioutput_mean_squared_error,
    multioutput_precision_score,
    multioutput_recall_score,
    multioutput_root_mean_squared_error,
)


@pytest.fixture
def metrics_list() -> list[tuple[str, Callable, Callable]]:
    return [
        ("Accuracy", accuracy_score, multioutput_accuracy_score),
        ("AUPRC", average_precision_score, multioutput_auprc_score),
        ("AUROC", roc_auc_score, multioutput_auroc_score),
        (
            "Balanced accuracy",
            balanced_accuracy_score,
            multioutput_balanced_accuracy_score,
        ),
        ("Cohen cappa", cohen_kappa_score, multioutput_cohen_kappa_score),
        ("F1", f1_score, multioutput_f1_score),
        ("MCC", matthews_corrcoef, multioutput_matthews_corr_coef),
        ("MAE", mean_absolute_error, multioutput_mean_absolute_error),
        ("MSE", mean_squared_error, multioutput_mean_squared_error),
        ("Precision", precision_score, multioutput_precision_score),
        ("Recall", recall_score, multioutput_recall_score),
        ("RMSE", root_mean_squared_error, multioutput_root_mean_squared_error),
    ]


def test_multioutput_metrics_single_task_equivalence(metrics_list):
    y_true = np.concatenate((np.ones(50), np.zeros(50)))
    y_pred = np.concatenate((np.ones(25), np.zeros(75)))

    for name, binary_metric, multioutput_metric in metrics_list:
        binary_value = binary_metric(y_true, y_pred)
        multioutput_value = multioutput_metric(y_true, y_pred)
        if not np.isclose(binary_value, multioutput_value):
            raise AssertionError(
                f"{name}, "
                f"metrics value differ:"
                f"binary {binary_value:.2f}, "
                f"multioutput {multioutput_value:.2f}"
            )


def test_multioutput_metrics_nan_present(metrics_list):
    # fmt: off
    y_true = np.array([
        [0, np.nan, 1],
        [0, 1, 0],
        [1, 0, np.nan]
    ])
    y_pred = np.array([
        [0, 0, 1],
        [0, 0, 1],
        [1, 0, 0]
    ])
    # fmt: on

    for name, binary_metric, multioutput_metric in metrics_list:
        # should not throw any errors
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            multioutput_metric(y_true, y_pred)


def test_multioutput_metrics_constant_columns(metrics_list):
    # fmt: off
    y_true = np.array([
        [0, 0, 1],
        [0, 1, 1],
        [0, 0, 1],
    ])
    y_pred = np.array([
        [0, 0, 1],
        [0, 0, 1],
        [1, 0, 0]
    ])
    # fmt: on

    for name, binary_metric, multioutput_metric in metrics_list:
        # should not throw any errors
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            multioutput_metric(y_true, y_pred)
