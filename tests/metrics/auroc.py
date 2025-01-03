from importlib.metadata import version

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from skfp.metrics import auroc_score


def test_auroc_score_sklearn_behavior():
    if _get_sklearn_version() >= 1.6:
        return

    y_true = np.array([1, 0, 0])
    y_score = np.array([0.7, 0.1, 0.05])
    sklearn_auroc = roc_auc_score(y_true, y_score)
    skfp_auroc = auroc_score(y_true, y_score)
    assert np.isclose(sklearn_auroc, skfp_auroc)


def test_auroc_score_y_true_constant_nan():
    if _get_sklearn_version() >= 1.6:
        return

    y_true = np.array([0, 0, 0])
    y_score = np.array([0.7, 0.1, 0.05])
    skfp_auroc = auroc_score(y_true, y_score)
    assert np.isnan(skfp_auroc)

    y_true = np.array([1, 1, 1])
    skfp_auroc = auroc_score(y_true, y_score)
    assert np.isnan(skfp_auroc)


def test_auroc_score_can_raise_error():
    if _get_sklearn_version() >= 1.6:
        return

    y_true = np.array([0, 0, 0])
    y_score = np.array([0.7, 0.1, 0.05])
    with pytest.raises(ValueError) as exc_info:
        auroc_score(y_true, y_score, constant_target_behavior="raise")
    assert "Only one class present in y_true" in str(exc_info)

    y_true = np.array([1, 1, 1])
    with pytest.raises(ValueError) as exc_info:
        auroc_score(y_true, y_score, constant_target_behavior="raise")
    assert "Only one class present in y_true" in str(exc_info)


def test_auroc_score_y_true_constant_float():
    if _get_sklearn_version() >= 1.6:
        return

    y_true = np.array([0, 0, 0])
    y_score = np.array([0.7, 0.1, 0.05])
    skfp_auroc = auroc_score(y_true, y_score, constant_target_behavior=0.5)
    assert np.isclose(skfp_auroc, 0.5)

    skfp_auroc = auroc_score(y_true, y_score, constant_target_behavior=1.0)
    assert np.isclose(skfp_auroc, 1.0)


def _get_sklearn_version() -> float:
    str_ver = version("scikit-learn")  # e.g. 1.6.0
    str_ver = ".".join(str_ver.split(".")[:-1])  # e.g. 1.6
    return float(str_ver)
