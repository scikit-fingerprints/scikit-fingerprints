import warnings
from typing import Callable

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    make_scorer,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV

from skfp.metrics import (
    extract_multioutput_pos_proba,
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
    multioutput_spearman_correlation,
    spearman_correlation,
)


def get_all_metrics() -> list[tuple[str, Callable, Callable]]:
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
        (
            "Spearman correlation",
            spearman_correlation,
            multioutput_spearman_correlation,
        ),
    ]


@pytest.mark.parametrize(
    "metric_name, single_task_metric, multioutput_metric",
    get_all_metrics(),
)
def test_multioutput_metrics_single_task_equivalence(
    metric_name, single_task_metric, multioutput_metric
):
    y_true = np.concatenate((np.ones(50), np.zeros(50)))
    y_pred = np.concatenate((np.ones(25), np.zeros(75)))

    single_task_value = single_task_metric(y_true, y_pred)
    multioutput_value = multioutput_metric(y_true, y_pred)
    if not np.isclose(single_task_value, multioutput_value):
        raise AssertionError(
            f"{metric_name} values differ:"
            f"single-task {single_task_value:.2f}, "
            f"multioutput {multioutput_value:.2f}"
        )


@pytest.mark.parametrize(
    "metric_name, single_task_metric, multioutput_metric",
    get_all_metrics(),
)
def test_multioutput_metrics_nan_present(
    metric_name, single_task_metric, multioutput_metric
):
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

    # should not throw any errors
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            multioutput_metric(y_true, y_pred)
        except Exception as e:
            raise ValueError(
                f"{metric_name} raised an error with NaNs present, error:\n{str(e)}"
            )


@pytest.mark.parametrize(
    "metric_name, single_task_metric, multioutput_metric",
    get_all_metrics(),
)
def test_multioutput_metrics_constant_columns(
    metric_name, single_task_metric, multioutput_metric
):
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

    # should not throw any errors
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            multioutput_metric(y_true, y_pred)
        except Exception as e:
            raise ValueError(
                f"{metric_name} raised an error with constant columns present, "
                f"error:\n{str(e)}"
            )


@pytest.mark.parametrize(
    "metric_name, single_task_metric, multioutput_metric",
    get_all_metrics(),
)
def test_multioutput_different_shapes(
    metric_name, single_task_metric, multioutput_metric
):
    y_true = np.ones((10, 2))
    y_pred = np.zeros((10, 2))

    # should not throw any errors
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            multioutput_metric(y_true, y_pred)
        except Exception as e:
            raise ValueError(
                f"{metric_name} raised an error with same y_true and y_pred shape, "
                f"error:\n{str(e)}"
            )

    y_true = np.ones(10)
    y_pred = np.zeros((10, 2))

    # should always throw an error
    with pytest.raises(
        ValueError,
        match="Both true labels and predictions must have the same shape",
    ):
        multioutput_metric(y_true, y_pred)


@pytest.mark.parametrize(
    "metric_name, single_task_metric, multioutput_metric",
    get_all_metrics(),
)
def test_multioutput_metrics_grid_search_compatible(
    metric_name, single_task_metric, multioutput_metric
):
    n_samples = 100
    n_tasks = 5

    # regression also works on this
    X, _ = make_classification(
        n_samples=n_samples, n_features=10, n_classes=2, random_state=0
    )
    y = np.random.randint(low=0, high=2, size=(n_samples, n_tasks), dtype=int)

    if metric_name in {"MAE", "MSE", "RMSE"}:
        estimator = RandomForestRegressor()
    else:
        estimator = RandomForestClassifier()

    response_method = (
        "predict_proba" if metric_name in {"AUPRC", "AUROC"} else "predict"
    )
    greater_is_better = False if metric_name in {"MAE", "MSE", "RMSE"} else True

    scorer = make_scorer(
        multioutput_metric,
        response_method=response_method,
        greater_is_better=greater_is_better,
    )

    # should not throw any errors
    cv = GridSearchCV(
        estimator=estimator,
        param_grid={"n_estimators": [5, 10]},
        scoring=scorer,
    )
    cv.fit(X, y)


def test_extract_multioutput_pos_proba():
    n_samples = 10
    n_tasks = 5

    predictions = [np.random.rand(n_samples, 2) for _ in range(n_tasks)]
    predictions = extract_multioutput_pos_proba(predictions)

    assert predictions.shape == (n_samples, n_tasks)
