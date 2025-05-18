import numpy as np
import pytest

from skfp.applicability_domain import HotellingT2TestADChecker
from tests.applicability_domain.utils import get_data_inside_ad, get_data_outside_ad


def test_inside_hotelling_t2_test_ad():
    X_train, X_test = get_data_inside_ad()

    ad_checker = HotellingT2TestADChecker()
    ad_checker.fit(X_train)

    scores = ad_checker.score_samples(X_test)
    assert np.all(scores >= 0)

    preds = ad_checker.predict(X_test)
    assert isinstance(preds, np.ndarray)
    assert np.isdtype(preds.dtype, np.bool)
    assert preds.shape == (len(X_test),)
    assert np.all(preds == 1)


def test_outside_hotelling_t2_test_ad():
    X_train, X_test = get_data_outside_ad()

    ad_checker = HotellingT2TestADChecker()
    ad_checker.fit(X_train)

    scores = ad_checker.score_samples(X_test)
    assert np.all(scores >= 0)

    preds = ad_checker.predict(X_test)
    assert isinstance(preds, np.ndarray)
    assert np.isdtype(preds.dtype, np.bool)
    assert preds.shape == (len(X_test),)
    assert np.all(preds == 0)


@pytest.mark.parametrize("alpha", np.linspace(0.1, 0.25))
def test_hotelling_t2_test_options(alpha):
    # everything should work for all options
    X_train, X_test = get_data_inside_ad()

    ad_checker = HotellingT2TestADChecker(alpha)
    ad_checker.fit(X_train)

    scores = ad_checker.score_samples(X_test)
    assert np.all(scores >= 0)

    preds = ad_checker.predict(X_test)
    assert isinstance(preds, np.ndarray)
    assert np.isdtype(preds.dtype, np.bool)
    assert preds.shape == (len(X_test),)
    assert np.all(preds == 1)


def test_hotelling_t2_test_pass_y_train():
    # smoke test, should not throw errors
    X_train, _ = get_data_inside_ad()
    y_train = np.zeros(len(X_train))
    ad_checker = HotellingT2TestADChecker()
    ad_checker.fit(X_train, y_train)
