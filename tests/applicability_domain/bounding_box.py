import numpy as np

from skfp.applicability_domain import BoundingBoxADChecker
from tests.applicability_domain.utils import get_data_inside_ad, get_data_outside_ad


def test_inside_ad():
    X_train, X_test = get_data_inside_ad()

    ad_checker = BoundingBoxADChecker()
    ad_checker.fit(X_train)

    scores = ad_checker.score_samples(X_test)
    assert np.all(scores >= 0)

    preds = ad_checker.predict(X_test)
    assert isinstance(preds, np.ndarray)
    assert np.isdtype(preds.dtype, np.bool)
    assert preds.shape == (len(X_test),)
    assert np.all(preds == 1)


def test_mols_outside_ad():
    X_train, X_test = get_data_outside_ad()

    ad_checker = BoundingBoxADChecker()
    ad_checker.fit(X_train)

    scores = ad_checker.score_samples(X_test)
    assert np.all(scores >= 0)

    preds = ad_checker.predict(X_test)
    assert isinstance(preds, np.ndarray)
    assert np.isdtype(preds.dtype, np.bool)
    assert preds.shape == (len(X_test),)
    assert np.all(preds == 0)


def test_three_sigma():
    X_train, X_test = get_data_inside_ad()
    ad_checker = BoundingBoxADChecker(
        percentile_lower="three_sigma", percentile_upper="three_sigma"
    )
    ad_checker.fit(X_train)
    preds = ad_checker.predict(X_test)
    assert np.all(preds == 1)


def test_lower_percentages():
    X_train, _ = get_data_inside_ad()
    X_test = X_train + 10

    # looser check, AD is larger
    ad_checker = BoundingBoxADChecker(percentile_lower=0)
    ad_checker.fit(X_train)
    avg_score_0 = np.mean(ad_checker.score_samples(X_test))

    # stricter check, AD is smaller
    ad_checker = BoundingBoxADChecker(percentile_lower=25)
    ad_checker.fit(X_train)
    avg_score_25 = np.mean(ad_checker.score_samples(X_test))

    # average inlier score should be greater for larger AD
    assert avg_score_0 > avg_score_25


def test_num_allowed_violations():
    X_train, X_test = get_data_inside_ad(n_train=100, n_test=100)
    X_test += 25

    ad_checker = BoundingBoxADChecker(num_allowed_violations=0)
    ad_checker.fit(X_train)
    preds = ad_checker.predict(X_test)
    assert not np.all(preds == 1)

    ad_checker = BoundingBoxADChecker(num_allowed_violations=1)
    ad_checker.fit(X_train)
    preds = ad_checker.predict(X_test)
    assert np.all(preds == 1)


def test_pass_y_train():
    # smoke test, should not throw errors
    X_train = np.vstack((np.zeros((10, 5)), np.ones((10, 5))))
    y_train = np.zeros(10)
    ad_checker = BoundingBoxADChecker()
    ad_checker.fit(X_train, y_train)
