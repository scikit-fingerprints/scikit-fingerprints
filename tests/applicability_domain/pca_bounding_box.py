import numpy as np
import pytest
from numpy.testing import assert_equal

from skfp.applicability_domain import PCABoundingBoxADChecker
from tests.applicability_domain.utils import get_data_inside_ad, get_data_outside_ad


def test_inside_pca_ad():
    X_train, X_test = get_data_inside_ad()

    ad_checker = PCABoundingBoxADChecker()
    ad_checker.fit(X_train)

    scores = ad_checker.score_samples(X_test)
    assert np.all(scores >= 0)

    preds = ad_checker.predict(X_test)
    assert isinstance(preds, np.ndarray)
    assert np.isdtype(preds.dtype, np.bool)
    assert_equal(preds.shape, (len(X_test),))
    assert np.all(preds == 1)


def test_outside_pca_ad():
    X_train, X_test = get_data_outside_ad()

    ad_checker = PCABoundingBoxADChecker()
    ad_checker.fit(X_train)

    scores = ad_checker.score_samples(X_test)
    assert np.all(scores >= 0)

    preds = ad_checker.predict(X_test)
    assert isinstance(preds, np.ndarray)
    assert np.isdtype(preds.dtype, np.bool)
    assert_equal(preds.shape, (len(X_test),))
    assert np.all(preds == 0)


@pytest.mark.parametrize(
    "n_components, whiten, random_state",
    [
        (n_components, whiten, random_state)
        for n_components in [1, 2, 0.5, 2, "mle", None]
        for whiten in [False, True]
        for random_state in [None, 0]
    ],
)
def test_pca_options(n_components, whiten, random_state):
    # everything should work for all options
    X_train, X_test = get_data_inside_ad()

    ad_checker = PCABoundingBoxADChecker(n_components, whiten, random_state)
    ad_checker.fit(X_train)

    scores = ad_checker.score_samples(X_test)
    assert np.all(scores >= 0)

    preds = ad_checker.predict(X_test)
    assert isinstance(preds, np.ndarray)
    assert np.isdtype(preds.dtype, np.bool)
    assert_equal(preds.shape, (len(X_test),))
    assert np.all(preds == 1)


def test_pca_pass_y_train():
    # smoke test, should not throw errors
    X_train, _ = get_data_inside_ad()
    y_train = np.zeros(len(X_train))
    ad_checker = PCABoundingBoxADChecker()
    ad_checker.fit(X_train, y_train)
