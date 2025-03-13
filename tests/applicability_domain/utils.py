import numpy as np
from sklearn.datasets import make_blobs


def get_data_inside_ad() -> tuple[np.ndarray, np.ndarray]:
    X_train, y_train = make_blobs(
        centers=1, cluster_std=10, center_box=(-10, 10), random_state=0
    )
    X_test, y_test = make_blobs(
        centers=1, cluster_std=0.1, center_box=(-0.1, 0.1), random_state=0
    )
    return X_train, X_test


def get_data_outside_ad() -> tuple[np.ndarray, np.ndarray]:
    X_train, y_train = make_blobs()
    X_test = X_train + 100
    return X_train, X_test
