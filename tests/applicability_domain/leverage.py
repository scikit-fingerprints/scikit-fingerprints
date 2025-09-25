import numpy as np
from numpy.testing import assert_equal

from skfp.applicability_domain import LeverageADChecker
from tests.applicability_domain.utils import get_data_inside_ad, get_data_outside_ad


def test_inside_leverage_ad():
    X_train, X_test = get_data_inside_ad()

    ad_checker = LeverageADChecker()
    ad_checker.fit(X_train)

    scores = ad_checker.score_samples(X_test)
    assert np.all(scores >= 0)

    preds = ad_checker.predict(X_test)
    assert isinstance(preds, np.ndarray)
    assert np.isdtype(preds.dtype, np.bool)
    assert_equal(preds.shape, (len(X_test),))
    assert np.all(preds == 1)


def test_outside_leverage_ad():
    X_train, X_test = get_data_outside_ad()

    ad_checker = LeverageADChecker()
    ad_checker.fit(X_train)

    scores = ad_checker.score_samples(X_test)
    assert np.all(scores >= 0)

    preds = ad_checker.predict(X_test)
    assert isinstance(preds, np.ndarray)
    assert np.isdtype(preds.dtype, np.bool)
    assert_equal(preds.shape, (len(X_test),))
    assert np.all(preds == 0)


def test_leverage_lower_threshold():
    X_train, _ = get_data_inside_ad()
    X_test = X_train + 10

    # looser check, AD is larger
    ad_checker = LeverageADChecker(threshold="auto")
    ad_checker.fit(X_train)
    num_passing_auto = np.sum(ad_checker.predict(X_test))

    # stricter check, AD is smaller
    ad_checker = LeverageADChecker(threshold=1e-5)
    ad_checker.fit(X_train)
    num_passing_small = np.sum(ad_checker.predict(X_test))

    # there should be more inliers for larger AD
    assert num_passing_auto > num_passing_small


def test_leverage_pass_y_train():
    # smoke test, should not throw errors
    X_train, _ = get_data_inside_ad()
    y_train = np.zeros(len(X_train))
    ad_checker = LeverageADChecker()
    ad_checker.fit(X_train, y_train)
