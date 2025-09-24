import numpy as np
import pytest
from numpy.testing import assert_equal
from sklearn.datasets import (
    make_blobs,
    make_classification,
    make_multilabel_classification,
    make_regression,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.utils._param_validation import InvalidParameterError

from skfp.applicability_domain import ProbStdADChecker


def test_inside_probstd_ad():
    X, y = make_blobs(
        n_samples=1000, centers=2, n_features=2, cluster_std=1.0, random_state=42
    )
    y = y.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    ad_checker = ProbStdADChecker(model=model, threshold=0.2)
    ad_checker.fit()

    scores = ad_checker.score_samples(X_test)
    assert np.all(scores >= 0)
    assert np.all(scores <= 0.5)

    preds = ad_checker.predict(X_test)
    assert isinstance(preds, np.ndarray)
    assert np.issubdtype(preds.dtype, np.bool_)
    assert_equal(preds.shape, (len(X_test),))
    assert np.all(preds == 1)


def test_outside_probstd_ad():
    np.random.seed(42)
    X = np.random.uniform(low=-5, high=5, size=(1000, 5))
    y = np.random.choice([0.0, 1.0], size=1000)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    ad_checker = ProbStdADChecker(model=model, threshold=0.05)
    ad_checker.fit()

    scores = ad_checker.score_samples(X_test)
    assert np.all(scores >= 0)
    assert np.all(scores <= 0.5)

    preds = ad_checker.predict(X_test)
    assert isinstance(preds, np.ndarray)
    assert np.issubdtype(preds.dtype, np.bool_)
    assert_equal(preds.shape, (len(X_test),))
    assert np.all(preds == 0)


def test_probstd_ad_with_default_model():
    X = np.random.uniform(size=(100, 5))
    y = X.sum(axis=1)

    ad_checker = ProbStdADChecker(threshold=0.2)
    ad_checker.fit(X, y)

    scores = ad_checker.score_samples(X)
    assert np.all(scores >= 0)
    assert np.all(scores <= 0.5)

    preds = ad_checker.predict(X)
    assert isinstance(preds, np.ndarray)
    assert np.isdtype(preds.dtype, np.bool)
    assert_equal(preds.shape, (len(X),))


def test_ptobstd_ad_checker_with_classifier():
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=42,
    )

    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)

    ad_checker = ProbStdADChecker(model=model, threshold=0.2)
    ad_checker.fit()

    scores = ad_checker.score_samples(X)
    assert scores.shape == (len(X),)
    assert np.all(scores >= 0)
    assert np.all(scores <= 0.5)

    preds = ad_checker.predict(X)
    assert isinstance(preds, np.ndarray)
    assert np.isdtype(preds.dtype, np.bool)
    assert_equal(preds.shape, (len(X),))


def test_probstd_fit():
    # smoke test, should not throw errors
    X_train, y_train = make_regression(
        n_samples=100, n_features=2, noise=0.1, random_state=42
    )

    rf = RandomForestRegressor(n_estimators=5, random_state=42)
    rf.fit(X_train, y_train)

    ad_checker = ProbStdADChecker(model=rf)
    ad_checker.fit(None, None)


def test_probstd_ad_checker_raise_error_on_multilabel():
    X, y = make_multilabel_classification(
        n_samples=50, n_features=5, n_classes=3, random_state=42
    )
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)

    ad_checker = ProbStdADChecker(model=model)

    with pytest.raises(InvalidParameterError, match="only supports binary classifiers"):
        ad_checker.fit()


def test_probstd_ad_checker_raise_error_on_multiclass():
    X, y = make_classification(
        n_samples=50, n_features=5, n_classes=3, n_clusters_per_class=1, random_state=42
    )
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)

    ad_checker = ProbStdADChecker(model=model)

    with pytest.raises(InvalidParameterError, match="only supports binary classifiers"):
        ad_checker.fit()


def test_probstd_ad_checker_raise_error_no_predict_proba():
    X, y = make_classification(n_samples=50, n_features=5, n_classes=2, random_state=42)
    model = LinearSVC()
    model.fit(X, y)

    ad_checker = ProbStdADChecker(model=model)

    with pytest.raises(InvalidParameterError) as exc_info:
        ad_checker.fit()

    assert "requires classifiers with .predict_proba() method" in str(exc_info)
