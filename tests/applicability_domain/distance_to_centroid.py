import numpy as np
import pytest
from scipy.spatial.distance import euclidean

from skfp.applicability_domain import DistanceToCentroidADChecker
from skfp.distances import ct4_count_distance
from tests.applicability_domain.utils import get_data_inside_ad, get_data_outside_ad


def test_inside_distance_to_centroid_ad():
    X_train, X_test = get_data_inside_ad()

    ad_checker = DistanceToCentroidADChecker()
    ad_checker.fit(X_train)

    scores = ad_checker.score_samples(X_test)
    assert np.all(scores >= 0)

    preds = ad_checker.predict(X_test)
    assert isinstance(preds, np.ndarray)
    assert np.isdtype(preds.dtype, np.bool)
    assert preds.shape == (len(X_test),)
    assert np.all(preds == 1)


def test_outside_distance_to_centroid_ad():
    X_train, X_test = get_data_outside_ad()

    ad_checker = DistanceToCentroidADChecker()
    ad_checker.fit(X_train)

    scores = ad_checker.score_samples(X_test)
    assert np.all(scores >= 0)

    preds = ad_checker.predict(X_test)
    assert isinstance(preds, np.ndarray)
    assert np.isdtype(preds.dtype, np.bool)
    assert preds.shape == (len(X_test),)
    assert np.all(preds == 0)


@pytest.mark.parametrize(
    "distance",
    [
        "euclidean",
        euclidean,
        "ct4_count_distance",
        ct4_count_distance,
        "bulk_ct4_count_distance",
    ],
)
def test_distance_functions(distance):
    X_train, X_test = get_data_inside_ad()

    # smoke test, should not throw errors
    ad_checker = DistanceToCentroidADChecker(distance=distance)
    ad_checker.fit(X_train)
    ad_checker.predict(X_test)


def test_distance_to_centroid_pass_y_train():
    # smoke test, should not throw errors
    X_train, _ = get_data_inside_ad()
    y_train = np.zeros(len(X_train))
    ad_checker = DistanceToCentroidADChecker()
    ad_checker.fit(X_train, y_train)
