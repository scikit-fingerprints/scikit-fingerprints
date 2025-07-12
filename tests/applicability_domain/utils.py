import numpy as np
from sklearn.datasets import make_blobs


def get_data_inside_ad(
    n_train: int = 1000, n_test: int = 100, binarize: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    n_features = 256 if binarize else 2

    X_train, _ = make_blobs(
        n_samples=n_train,
        centers=1,
        cluster_std=10.0,
        center_box=(-10.0, 10.0),
        random_state=0,
        n_features=n_features,
    )

    X_test, _ = make_blobs(
        n_samples=n_test,
        centers=1,
        cluster_std=0.1,
        center_box=(-0.1, 0.1),
        random_state=0,
        n_features=n_features,
    )

    return X_train, X_test


def get_data_outside_ad(
    n_train: int = 1000, n_test: int = 100, binarize: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    n_features = 256 if binarize else 2

    X_train, _ = make_blobs(
        n_samples=n_train,
        centers=1,
        cluster_std=10.0,
        center_box=(-10.0, 10.0),
        random_state=0,
        n_features=n_features,
    )

    if binarize:
        thresholds = np.median(X_train, axis=0)
        X_train = (X_train > thresholds).astype(int)
        X_test = 1 - X_train[:n_test]
    else:
        X_test = X_train[:n_test] + 100.0

    return X_train, X_test
