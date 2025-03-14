import numpy as np
from sklearn.datasets import make_blobs


def get_data_inside_ad(
    n_train: int = 1000, n_test: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    X_train, y_train = make_blobs(
        n_samples=n_train,
        n_features=10,
        centers=1,
        cluster_std=10,
        center_box=(-10, 10),
        random_state=0,
    )
    X_test, y_test = make_blobs(
        n_samples=n_test,
        centers=1,
        n_features=10,
        cluster_std=0.1,
        center_box=(-0.1, 0.1),
        random_state=0,
    )
    return X_train, X_test


def get_data_outside_ad(
    n_train: int = 1000, n_test: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    X_train, y_train = make_blobs(n_samples=n_train)
    X_test = X_train[:n_test] + 100
    return X_train, X_test
