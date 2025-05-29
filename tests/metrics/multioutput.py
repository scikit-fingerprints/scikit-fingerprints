from collections.abc import Callable

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
from skfp.metrics.multioutput import _safe_multioutput_metric


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
def test_multioutput_metrics_valid_input(
    metric_name, single_task_metric, multioutput_metric
):
    # fmt: off
    y_true = np.array([
        [0, 1, 1],
        [0, 1, 0],
        [1, 0, 0]
    ])
    y_pred = np.array([
        [0, 0, 1],
        [0, 1, 1],
        [1, 0, 0]
    ])
    # fmt: on

    # should not throw any errors
    multioutput_metric(y_true, y_pred, suppress_warnings=True)


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
    multioutput_metric(y_true, y_pred, suppress_warnings=True)


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
        [0, 1, 0],
        [0, 0, 1],
    ])
    y_pred = np.array([
        [0, 0, 1],
        [0, 0, 1],
        [1, 0, 0]
    ])
    # fmt: on

    # should not throw any errors
    multioutput_metric(y_true, y_pred, suppress_warnings=True)


def test_multioutput_spearman_correlation_constant_columns():
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

    # Spearman correlation should throw an error, since one column is always
    # constant, and it returns NaN in those cases, resulting in 3 NaN values
    with pytest.raises(ValueError, match="Could not compute metric.*"):
        multioutput_spearman_correlation(y_true, y_pred, suppress_warnings=True)


@pytest.mark.parametrize(
    "metric_name, single_task_metric, multioutput_metric",
    get_all_metrics(),
)
def test_multioutput_different_shapes(
    metric_name, single_task_metric, multioutput_metric
):
    y_true = np.ones((10, 2))
    y_pred = np.zeros((10, 2))

    # ensure we have non-constant values to avoid errors
    y_true[0] = [0, 0]
    y_pred[0] = [1, 1]

    # should not throw any errors
    multioutput_metric(y_true, y_pred, suppress_warnings=True)

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


def test_list_conversion():
    # should convert internally and not throw any errors
    _safe_multioutput_metric(mean_squared_error, y_true=[1, 2, 3], y_pred=[1, 2, 3])


def test_skip_all_nan_column():
    # should ignore all-NaN column
    y_true = np.array(
        [
            [1, np.nan, 1],
            [2, np.nan, 1],
            [3, np.nan, 2],
        ]
    )
    y_pred = np.array(
        [
            [1, 2, 3],
            [1, 3, 4],
            [1, 3, 2],
        ]
    )
    score_with_nan = _safe_multioutput_metric(mean_squared_error, y_true, y_pred)
    score_without_nan = _safe_multioutput_metric(
        mean_squared_error, y_true[:, [0, 2]], y_pred[:, [0, 2]]
    )
    assert np.isclose(score_with_nan, score_without_nan)


def test_metrics_inputs_shapes():
    # 1D and 2D should simply work
    _safe_multioutput_metric(mean_squared_error, y_true=[1, 2, 3], y_pred=[1, 2, 3])
    _safe_multioutput_metric(mean_squared_error, y_true=[[1, 2, 3]], y_pred=[[1, 2, 3]])

    # different dimensions should throw an error
    with pytest.raises(ValueError) as error:
        _safe_multioutput_metric(
            mean_squared_error, y_true=[[[1, 2, 3]]], y_pred=[1, 2, 3]
        )

    assert "Both true labels and predictions must have the same shape" in str(error)
    assert "true labels (1, 1, 3), predictions (3,)" in str(error)

    # over 2 dimensions should throw an error
    with pytest.raises(ValueError) as error:
        _safe_multioutput_metric(
            mean_squared_error, y_true=[[[1, 2, 3]]], y_pred=[[[1, 2, 3]]]
        )

    assert "Both true labels and predictions must have 1 or 2 dimensions" in str(error)
    assert "true labels 3, predictions 3" in str(error)
