import numpy as np
from sklearn.datasets import make_blobs, make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

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

    scores = ad_checker.score_samples(X_test)
    assert np.all(scores >= 0)
    assert np.all(scores <= 0.5)

    preds = ad_checker.predict(X_test)
    assert isinstance(preds, np.ndarray)
    assert np.issubdtype(preds.dtype, np.bool_)
    assert preds.shape == (len(X_test),)

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

    scores = ad_checker.score_samples(X_test)
    assert np.all(scores >= 0)
    assert np.all(scores <= 0.5)

    preds = ad_checker.predict(X_test)
    assert isinstance(preds, np.ndarray)
    assert np.issubdtype(preds.dtype, np.bool_)
    assert preds.shape == (len(X_test),)

    assert np.all(preds == 0)


def test_probstd_fit():
    # smoke test, should not throw errors
    X_train, y_train = make_regression(
        n_samples=100, n_features=2, noise=0.1, random_state=42
    )

    rf = RandomForestRegressor(n_estimators=5, random_state=42)
    rf.fit(X_train, y_train)

    ad_checker = ProbStdADChecker(model=rf)
    ad_checker.fit(X_train, y_train)
