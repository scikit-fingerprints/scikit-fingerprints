import numpy as np
import pytest
from numpy.testing import assert_equal

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
    assert np.issubdtype(preds.dtype, np.bool_)
    assert_equal(preds.shape, (len(y_test),))
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
    assert np.issubdtype(preds.dtype, np.bool_)
    assert_equal(preds.shape, (len(y_test),))
    assert np.all(preds == 0)


def test_response_range_with_threshold():
    y_train = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
    y_test_in = np.array([0.95, 1.0, 1.05])
    y_test_out = np.array([0.85, 1.15])

    ad_checker = ResponseVariableRangeADChecker(threshold=0.1)
    ad_checker.fit(y_train)

    preds_in = ad_checker.predict(y_test_in)
    preds_out = ad_checker.predict(y_test_out)

    scores_in = ad_checker.score_samples(y_test_in)
    scores_out = ad_checker.score_samples(y_test_out)
    assert np.all(scores_in >= 0)
    assert np.all(scores_out >= 0)

    assert np.all(preds_in == 1)
    assert np.all(preds_out == 0)


def test_response_range_raises_on_multilabel():
    y = np.random.uniform(low=0.0, high=1.0, size=(100, 2))
    ad_checker = ResponseVariableRangeADChecker()

    with pytest.raises(ValueError, match="only supports 1D target values"):
        ad_checker.fit(y)


def test_response_range_pass_y_train():
    # smoke test, should not throw errors
    X_train = np.vstack((np.zeros((10, 5)), np.ones((10, 5))))
    y_train = np.zeros(10)
    ad_checker = ResponseVariableRangeADChecker()
    ad_checker.fit(y_train, X_train)
