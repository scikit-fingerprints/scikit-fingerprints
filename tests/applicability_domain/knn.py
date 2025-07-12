import numpy as np
import pytest

from skfp.applicability_domain import KNNADChecker
from tests.applicability_domain.utils import get_data_inside_ad, get_data_outside_ad

ALLOWED_METRICS = [
    "tanimoto_binary",
    "tanimoto_count",
]

ALLOWED_AGGS = ["mean", "max", "min"]


@pytest.mark.parametrize("metric", ALLOWED_METRICS)
@pytest.mark.parametrize("agg", ALLOWED_AGGS)
def test_inside_knn_ad(metric, agg):
    if metric == "tanimoto_binary":
        X_train, X_test = get_data_inside_ad(binarize=True)
    else:
        X_train, X_test = get_data_inside_ad()

    ad_checker = KNNADChecker(k=3, agg=agg)
    ad_checker.fit(X_train)

    scores = ad_checker.score_samples(X_test)
    assert scores.shape == (len(X_test),)

    preds = ad_checker.predict(X_test)
    assert isinstance(preds, np.ndarray)
    assert np.issubdtype(preds.dtype, np.bool_)
    assert preds.shape == (len(X_test),)
    assert np.all(preds == 1)


@pytest.mark.parametrize("metric", ALLOWED_METRICS)
@pytest.mark.parametrize("agg", ALLOWED_AGGS)
def test_outside_knn_ad(metric, agg):
    if metric == "tanimoto_binary":
        X_train, X_test = get_data_outside_ad(binarize=True)
    else:
        X_train, X_test = get_data_outside_ad()

    ad_checker = KNNADChecker(k=3, metric=metric, agg=agg)
    ad_checker.fit(X_train)

    scores = ad_checker.score_samples(X_test)
    assert np.all(scores >= 0)

    preds = ad_checker.predict(X_test)
    print(preds)
    assert isinstance(preds, np.ndarray)
    assert np.issubdtype(preds.dtype, np.bool_)
    assert preds.shape == (len(X_test),)
    assert np.all(preds == 0)


@pytest.mark.parametrize("metric", ALLOWED_METRICS)
@pytest.mark.parametrize("agg", ALLOWED_AGGS)
def test_knn_different_k_values(metric, agg):
    if metric == "tanimoto_binary":
        X_train, X_test = get_data_inside_ad(binarize=True)
    else:
        X_train, X_test = get_data_inside_ad()

    # smaller k, stricter check
    ad_checker_k1 = KNNADChecker(k=1, metric=metric, agg=agg)
    ad_checker_k1.fit(X_train)
    passed_k1 = ad_checker_k1.predict(X_test).sum()

    # larger k, potentially less strict
    ad_checker_k5 = KNNADChecker(k=5, metric=metric, agg=agg)
    ad_checker_k5.fit(X_train)
    passed_k5 = ad_checker_k5.predict(X_test).sum()

    # both should be valid results
    assert isinstance(passed_k1, (int, np.integer))
    assert isinstance(passed_k5, (int, np.integer))


@pytest.mark.parametrize("metric", ALLOWED_METRICS)
@pytest.mark.parametrize("agg", ALLOWED_AGGS)
def test_knn_pass_y_train(metric, agg):
    # smoke test, should not throw errors
    if metric == "tanimoto_binary":
        X_train, _ = get_data_inside_ad(binarize=True)
    else:
        X_train, _ = get_data_inside_ad()

    y_train = np.zeros(len(X_train))
    ad_checker = KNNADChecker(k=3, metric=metric, agg=agg)
    ad_checker.fit(X_train, y_train)


@pytest.mark.parametrize("metric", ["mean", "max"])
@pytest.mark.parametrize("agg", ALLOWED_AGGS)
def test_knn_invalid_k(metric, agg):
    if metric == "tanimoto_binary":
        X_train, _ = get_data_inside_ad(binarize=True)
    else:
        X_train, _ = get_data_inside_ad()

    with pytest.raises(
        ValueError,
        match=r"k \(\d+\) must be smaller than the number of training samples \(\d+\)",
    ):
        ad_checker = KNNADChecker(k=len(X_train), metric=metric, agg=agg)
        ad_checker.fit(X_train)


def test_knn_invalid_metric():
    X_train, _ = get_data_inside_ad()
    ad_checker = KNNADChecker(k=3, metric="euclidean")
    with pytest.raises(KeyError):
        ad_checker.fit(X_train)
