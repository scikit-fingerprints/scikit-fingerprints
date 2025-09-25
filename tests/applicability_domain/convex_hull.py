import numpy as np
from numpy.testing import assert_equal

from skfp.applicability_domain import ConvexHullADChecker
from tests.applicability_domain.utils import get_data_inside_ad, get_data_outside_ad


def test_inside_convex_hull_ad():
    X_train, X_test = get_data_inside_ad(n_train=100, n_test=10)

    ad_checker = ConvexHullADChecker()
    ad_checker.fit(X_train)

    scores = ad_checker.score_samples(X_test)
    assert np.all(scores == 1)

    preds = ad_checker.predict(X_test)
    assert isinstance(preds, np.ndarray)
    assert np.isdtype(preds.dtype, np.bool)
    assert_equal(preds.shape, (len(X_test),))
    assert np.all(preds == 1)


def test_outside_convex_hull_ad():
    X_train, X_test = get_data_outside_ad(n_train=100, n_test=10)

    ad_checker = ConvexHullADChecker()
    ad_checker.fit(X_train)

    scores = ad_checker.score_samples(X_test)
    assert np.all(scores == 0)

    preds = ad_checker.predict(X_test)
    assert isinstance(preds, np.ndarray)
    assert np.isdtype(preds.dtype, np.bool)
    assert_equal(preds.shape, (len(X_test),))
    assert np.all(preds == 0)


def test_convex_hull_pass_y_train():
    # smoke test, should not throw errors
    X_train = np.vstack((np.zeros((10, 5)), np.ones((10, 5))))
    y_train = np.zeros(10)
    ad_checker = ConvexHullADChecker()
    ad_checker.fit(X_train, y_train)
