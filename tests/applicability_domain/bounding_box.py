import numpy as np
from applicability_domain.utils import get_data_inside_ad, get_data_outside_ad

from skfp.applicability_domain import BoundingBoxADChecker


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


def test_pass_y_train():
    # smoke test, should not throw errors
    X_train = np.vstack((np.zeros((10, 5)), np.ones((10, 5))))
    y_train = np.zeros(10)
    ad_checker = BoundingBoxADChecker()
    ad_checker.fit(X_train, y_train)
