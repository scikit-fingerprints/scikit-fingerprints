import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from skfp.applicability_domain import StandardDeviationADChecker


def test_inside_std_ad():
    X_train, y_train = make_regression(
        n_samples=1000, n_features=2, noise=0.1, random_state=42
    )
    X_test = X_train[:100]

    rf = RandomForestRegressor(n_estimators=5, random_state=42)
    rf.fit(X_train, y_train)

    ad_checker = StandardDeviationADChecker(model=rf, threshold=10.0)

    scores = ad_checker.score_samples(X_test)
    assert np.all(scores >= 0)

    preds = ad_checker.predict(X_test)
    assert isinstance(preds, np.ndarray)
    assert np.isdtype(preds.dtype, np.bool)
    assert preds.shape == (len(X_test),)
    assert np.all(preds == 1)


def test_outside_std_ad():
    X_train, y_train = make_regression(
        n_samples=1000, n_features=2, noise=0.1, random_state=42
    )
    X_test = X_train[:100] * 100

    rf = RandomForestRegressor(n_estimators=5, random_state=42)
    rf.fit(X_train, y_train)

    ad_checker = StandardDeviationADChecker(model=rf, threshold=1.0)

    scores = ad_checker.score_samples(X_test)
    assert np.all(scores >= 0)

    preds = ad_checker.predict(X_test)
    assert isinstance(preds, np.ndarray)
    assert np.isdtype(preds.dtype, np.bool)
    assert preds.shape == (len(X_test),)
    assert np.all(preds == 0)


def test_std_ad_with_untrained_model():
    X = np.random.uniform(size=(100, 5))
    y = X.sum(axis=1)

    model = RandomForestRegressor(n_estimators=5, random_state=0)
    ad_checker = StandardDeviationADChecker(model=model, threshold=1.0)
    ad_checker.fit(X, y)

    scores = ad_checker.score_samples(X)
    assert np.all(scores >= 0)

    preds = ad_checker.predict(X)
    assert isinstance(preds, np.ndarray)
    assert np.isdtype(preds.dtype, np.bool)
    assert preds.shape == (len(X),)


def test_std_ad_with_default_model():
    X = np.random.uniform(size=(100, 5))
    y = X.sum(axis=1)

    ad_checker = StandardDeviationADChecker(threshold=1.0)
    ad_checker.fit(X, y)

    scores = ad_checker.score_samples(X)
    assert np.all(scores >= 0)

    preds = ad_checker.predict(X)
    assert isinstance(preds, np.ndarray)
    assert np.isdtype(preds.dtype, np.bool)
    assert preds.shape == (len(X),)


def test_standard_deviation_ad_checker_with_classifier():
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=42,
    )

    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)

    ad_checker = StandardDeviationADChecker(model=model, threshold=0.5)

    scores = ad_checker.score_samples(X)
    assert scores.shape == (len(X),)
    assert np.all(scores >= 0)

    preds = ad_checker.predict(X)
    assert isinstance(preds, np.ndarray)
    assert np.isdtype(preds.dtype, np.bool)
    assert preds.shape == (len(X),)


def test_std_fit():
    # smoke test, should not throw errors
    X_train, y_train = make_regression(
        n_samples=100, n_features=2, noise=0.1, random_state=42
    )

    rf = RandomForestRegressor(n_estimators=5, random_state=42)
    rf.fit(X_train, y_train)

    ad_checker = StandardDeviationADChecker(model=rf)
    ad_checker.fit(X_train, y_train)
