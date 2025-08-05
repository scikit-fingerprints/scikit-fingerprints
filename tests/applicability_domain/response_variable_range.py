import numpy as np

from skfp.applicability_domain import ResponseVariableRangeADChecker
from tests.applicability_domain.utils import get_data_inside_ad, get_data_outside_ad


def test_inside_response_range_ad():
    y_train, y_test = get_data_inside_ad()
    y_train = y_train[:, 0]
    y_test = y_test[:, 0]

    ad_checker = ResponseVariableRangeADChecker()
    ad_checker.fit(y_train)

    scores = ad_checker.score_samples(y_test)
    assert np.all(scores >= 0)

    preds = ad_checker.predict(y_test)
    assert isinstance(preds, np.ndarray)
    assert np.isdtype(preds.dtype, np.bool)
    assert preds.shape == (len(y_test),)
    assert np.all(preds == 1)


def test_outside_response_range_ad():
    y_train, y_test = get_data_outside_ad()
    y_train = y_train[:, 0]
    y_test = y_test[:, 0]

    ad_checker = ResponseVariableRangeADChecker()
    ad_checker.fit(y_train)

    scores = ad_checker.score_samples(y_test)
    assert np.all(scores >= 0)

    preds = ad_checker.predict(y_test)
    assert isinstance(preds, np.ndarray)
    assert np.isdtype(preds.dtype, np.bool)
    assert preds.shape == (len(y_test),)
    assert np.all(preds == 0)


def test_response_range_pass_y_train():
    # smoke test, should not throw errors
    X_train = np.vstack((np.zeros((10, 5)), np.ones((10, 5))))
    y_train = np.zeros(10)
    ad_checker = ResponseVariableRangeADChecker()
    ad_checker.fit(X_train, y_train)
