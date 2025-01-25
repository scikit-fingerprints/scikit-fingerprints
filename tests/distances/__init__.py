import numpy as np
import pytest
from scipy.sparse import csr_array
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from skfp.distances import (
    all_bit_binary_distance,
    all_bit_count_distance,
    dice_binary_distance,
    dice_count_distance,
    tanimoto_binary_distance,
    tanimoto_count_distance,
)
from skfp.distances.utils import _check_nan


def test_check_nan_numpy():
    vec_a = np.array([1, 2, np.nan, 4, 5])

    with pytest.raises(ValueError, match="Input array contains NaN values"):
        _check_nan(vec_a)


def test_check_nan_scipy():
    sparse_matrix = csr_array([[1, 2, np.nan, 4, 5]])

    with pytest.raises(ValueError, match="Input sparse matrix contains NaN values"):
        _check_nan(sparse_matrix)


def test_check_nan_wrong_type():
    with pytest.raises(TypeError) as exc_info:
        _check_nan(1)

    assert "Expected numpy.ndarray or scipy.sparse.csr_array" in str(exc_info)


@pytest.mark.parametrize(
    "method, vec_a, vec_b",
    [
        (dice_binary_distance, np.array([0, 1, 0, 1]), np.array([0, 1, 1, 0])),
        (dice_binary_distance, csr_array([[0, 1, 0, 1]]), csr_array([[0, 1, 1, 0]])),
        (tanimoto_binary_distance, np.array([0, 1, 0, 1]), np.array([0, 1, 1, 0])),
        (
            tanimoto_binary_distance,
            csr_array([[0, 1, 0, 1]]),
            csr_array([[0, 1, 1, 0]]),
        ),
        (all_bit_binary_distance, np.array([0, 1, 0, 1]), np.array([0, 1, 1, 0])),
        (all_bit_binary_distance, csr_array([[0, 1, 0, 1]]), csr_array([[0, 1, 1, 0]])),
    ],
)
def test_sklearn_pairwise_compatible_binary(method, vec_a, vec_b):
    if isinstance(vec_a, csr_array) and isinstance(vec_b, csr_array):
        sklearn_dist = pairwise_distances(vec_a, vec_b, metric=method)
        assert method(vec_a, vec_b) == sklearn_dist[0][0]
    else:
        vec_a = [vec_a]
        vec_b = [vec_b]
        sklearn_dist = pairwise_distances(vec_a, vec_b, metric=method)
        assert method(vec_a[0], vec_b[0]) == sklearn_dist[0][0]


@pytest.mark.parametrize(
    "method, vec_a, vec_b",
    [
        (dice_count_distance, np.array([1, 2, 3, 4]), np.array([1, 2, 3, 5])),
        (dice_count_distance, csr_array([[1, 2, 3, 4]]), csr_array([[1, 2, 3, 5]])),
        (tanimoto_count_distance, np.array([1, 2, 3, 4]), np.array([1, 2, 3, 5])),
        (tanimoto_count_distance, csr_array([[1, 2, 3, 4]]), csr_array([[1, 2, 3, 5]])),
        (all_bit_count_distance, np.array([1, 2, 3, 4]), np.array([1, 2, 3, 5])),
        (all_bit_count_distance, csr_array([[1, 2, 3, 4]]), csr_array([[1, 2, 3, 5]])),
    ],
)
def test_sklearn_pairwise_compatible_count(method, vec_a, vec_b):
    if isinstance(vec_a, csr_array) and isinstance(vec_b, csr_array):
        vec_a = csr_array(vec_a)
        vec_b = csr_array(vec_b)
        sklearn_dist = pairwise_distances(vec_a, vec_b, metric=method)
        assert method(vec_a, vec_b) == sklearn_dist[0][0]
    else:
        vec_a = [vec_a]
        vec_b = [vec_b]
        sklearn_dist = pairwise_distances(vec_a, vec_b, metric=method)
        assert method(vec_a[0], vec_b[0]) == sklearn_dist[0][0]


@pytest.mark.parametrize(
    "method, vec_a, vec_b",
    [
        (dice_binary_distance, np.array([1, 2, 3, 4]), np.array([1, 2, 3, 5])),
        (dice_binary_distance, csr_array([[1, 2, 3, 4]]), csr_array([[1, 2, 3, 5]])),
        (dice_count_distance, np.array([1, 2, 3, 4]), np.array([1, 2, 3, 5])),
        (dice_count_distance, csr_array([[1, 2, 3, 4]]), csr_array([[1, 2, 3, 5]])),
        (tanimoto_binary_distance, np.array([0, 1, 0, 1]), np.array([0, 1, 1, 0])),
        (
            tanimoto_binary_distance,
            csr_array([[0, 1, 0, 1]]),
            csr_array([[0, 1, 1, 0]]),
        ),
        (tanimoto_count_distance, np.array([1, 2, 3, 4]), np.array([1, 2, 3, 5])),
        (tanimoto_count_distance, csr_array([[1, 2, 3, 4]]), csr_array([[1, 2, 3, 5]])),
    ],
)
def test_sklearn_nearest_neighbors_compatible_binary(method, vec_a, vec_b):
    if isinstance(vec_a, csr_array) and isinstance(vec_b, csr_array):
        vec_a = csr_array(vec_a)
        vec_b = csr_array(vec_b)

        nn = NearestNeighbors(n_neighbors=1, metric=method)
        nn.fit(vec_a)
        sklearn_dist, _ = nn.kneighbors(vec_b)

        assert method(vec_a, vec_b) == sklearn_dist[0][0]
    else:
        vec_a = [vec_a]
        vec_b = [vec_b]

        nn = NearestNeighbors(n_neighbors=1, metric=method)
        nn.fit(vec_a)
        sklearn_dist, _ = nn.kneighbors(vec_b)

        assert method(vec_a[0], vec_b[0]) == sklearn_dist[0][0]


@pytest.mark.parametrize(
    "method",
    [
        dice_binary_distance,
        dice_count_distance,
        tanimoto_binary_distance,
        tanimoto_count_distance,
    ],
)
def test_different_types_raise_error(method, binary_numpy_array, binary_csr_array):
    with pytest.raises(TypeError) as exc_info:
        method(binary_numpy_array, binary_csr_array)

    assert "Both vec_a and vec_b must be of the same type," in str(exc_info)
    assert (
        "got <class 'numpy.ndarray'> and <class 'scipy.sparse._csr.csr_array'>"
        in str(exc_info)
    )
