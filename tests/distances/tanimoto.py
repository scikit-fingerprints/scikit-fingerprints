import numpy as np
import pytest
import sklearn
import sklearn.metrics
import sklearn.neighbors
from scipy import sparse
from scipy.sparse import csr_array

from skfp.distances.tanimoto import (
    _check_nan,
    tanimoto_binary_distance,
    tanimoto_binary_similarity,
    tanimoto_count_distance,
    tanimoto_count_similarity,
)


@pytest.fixture
def binary_numpy_array():
    vec = np.array([1, 0, 1, 0, 1])

    return vec


@pytest.fixture
def binary_csr_array():
    vec = csr_array([[1, 0, 1, 0, 1]])

    return vec


@pytest.mark.parametrize(
    "vec_a, vec_b, expected_similarity",
    [
        (np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]), 1.0),
        (np.array([1, 2, 3, 4]), np.array([0, 0, 0, 0]), 0.0),
        (np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0]), 1.0),
        (np.array([0, 1, 0, 0]), np.array([0, 0, 2, 3]), 0.0),
    ],
)
def test_tanimoto_count_numpy(vec_a, vec_b, expected_similarity):
    assert tanimoto_count_similarity(vec_a, vec_b) == expected_similarity


@pytest.mark.parametrize(
    "data_type, similarity_function",
    [
        (np.zeros, tanimoto_binary_similarity),
        (np.ones, tanimoto_binary_similarity),
        (np.zeros, tanimoto_count_similarity),
        (np.ones, tanimoto_count_similarity),
    ],
)
@pytest.mark.parametrize("matrix_type", ["numpy", "scipy"])
def test_tanimoto_similarity(data_type, similarity_function, matrix_type):
    size = 5
    if matrix_type == "numpy":
        vec_a = data_type(size, dtype=int)
        vec_b = data_type(size, dtype=int)
    elif matrix_type == "scipy":
        vec_a = sparse.csr_array(data_type((size, size), dtype=int))
        vec_b = sparse.csr_array(data_type((size, size), dtype=int))

    assert similarity_function(vec_a, vec_b) == 1.0


@pytest.mark.parametrize(
    "vec_a, vec_b, expected_comparison",
    [
        (np.array([1, 1, 0, 1]), np.array([1, 0, 1, 1]), "=="),
        (np.array([1, 1, 0, 1]), np.array([1, 1, 1, 1]), ">"),
        (np.array([1, 0, 0, 0]), np.array([1, 1, 1, 1]), "<"),
        (csr_array([[1, 1, 0, 1]]), csr_array([[1, 0, 1, 1]]), "=="),
        (csr_array([[1, 1, 0, 1]]), csr_array([[1, 1, 1, 1]]), ">"),
        (csr_array([[1, 0, 0, 0]]), csr_array([[1, 1, 1, 1]]), "<"),
    ],
)
def test_binary_against_threshold(vec_a, vec_b, expected_comparison):
    threshold = 0.5
    similarity = tanimoto_binary_similarity(vec_a, vec_b)

    if expected_comparison == "==":
        assert similarity == threshold
    elif expected_comparison == ">":
        assert similarity > threshold
    elif expected_comparison == "<":
        assert similarity < threshold


@pytest.mark.parametrize(
    "vec_a, vec_b, expected_comparison",
    [
        (np.array([1, 1, 0, 1]), np.array([1, 0, 1, 1]), "=="),
        (np.array([1, 0, 3, 0]), np.array([1, 0, 3, 0]), ">"),
        (np.array([1, 7, 3, 9]), np.array([1, 1, 6, 0]), "<"),
        (csr_array([[1, 1, 0, 1]]), csr_array([[1, 1, 0, 1]]), "=="),
        (csr_array([[1, 0, 3, 0]]), csr_array([[1, 0, 3, 0]]), ">"),
        (csr_array([[1, 7, 3, 9]]), csr_array([[1, 1, 6, 0]]), "<"),
    ],
)
def test_count_against_threshold(vec_a, vec_b, expected_comparison):
    threshold = 0.5
    similarity = tanimoto_count_similarity(vec_a, vec_b)

    if expected_comparison == ">":
        assert similarity > threshold
    elif expected_comparison == "<":
        assert similarity < threshold


@pytest.mark.parametrize(
    "vec_a, vec_b, expected_comparison",
    [
        (np.array([1, 1, 0, 1]), np.array([1, 0, 1, 1]), "=="),
        (np.array([1, 1, 0, 1]), np.array([1, 1, 1, 1]), "<"),
        (np.array([1, 0, 0, 0]), np.array([1, 1, 1, 1]), ">"),
        (csr_array([[1, 1, 0, 1]]), csr_array([[1, 1, 0, 1]]), "=="),
        (csr_array([[1, 1, 0, 1]]), csr_array([[1, 1, 1, 1]]), "<"),
        (csr_array([[1, 0, 0, 0]]), csr_array([[1, 1, 1, 1]]), ">"),
    ],
)
def test_binary_distance(
    vec_a,
    vec_b,
    expected_comparison,
):
    threshold = 0.5
    distance = tanimoto_binary_distance(vec_a, vec_b)

    if expected_comparison == "<":
        assert distance < threshold
    elif expected_comparison == ">":
        assert distance > threshold


@pytest.mark.parametrize(
    "vec_a, vec_b, expected_comparison",
    [
        (np.array([1, 1, 0, 1]), np.array([1, 0, 1, 1]), "=="),
        (np.array([1, 0, 3, 0]), np.array([1, 0, 3, 0]), "<"),
        (np.array([1, 9, 0, 0]), np.array([11, 1, 4, 5]), ">"),
        (csr_array([[1, 1, 0, 1]]), csr_array([[1, 1, 0, 1]]), "=="),
        (csr_array([[1, 0, 3, 0]]), csr_array([[1, 0, 3, 0]]), "<"),
        (csr_array([[1, 9, 0, 0]]), csr_array([[11, 1, 4, 5]]), ">"),
    ],
)
def test_count_distance(vec_a, vec_b, expected_comparison):
    threshold = 0.5
    distance = tanimoto_count_distance(vec_a, vec_b)

    if expected_comparison == "<":
        assert distance < threshold
    elif expected_comparison == ">":
        assert distance > threshold


@pytest.mark.parametrize(
    "vec_a, vec_b",
    [
        (np.array([0, 1, 0, 1]), np.array([0, 1, 1, 0])),
        (csr_array([[0, 1, 0, 1]]), csr_array([[0, 1, 1, 0]])),
    ],
)
def test_sklearn_pairwise_compatible_binary(vec_a, vec_b):
    if isinstance(vec_a, csr_array) and isinstance(vec_b, csr_array):
        sklearn_dist = sklearn.metrics.pairwise_distances(
            vec_a, vec_b, metric=tanimoto_binary_distance
        )
        assert tanimoto_binary_distance(vec_a, vec_b) == sklearn_dist[0][0]
    else:
        vec_a = [vec_a]
        vec_b = [vec_b]
        sklearn_dist = sklearn.metrics.pairwise_distances(
            vec_a, vec_b, metric=tanimoto_binary_distance
        )
        assert tanimoto_binary_distance(vec_a[0], vec_b[0]) == sklearn_dist[0][0]


@pytest.mark.parametrize(
    "vec_a, vec_b",
    [
        (np.array([1, 2, 3, 4]), np.array([1, 2, 3, 5])),
        (csr_array([[1, 2, 3, 4]]), csr_array([[1, 2, 3, 5]])),
    ],
)
def test_sklearn_pairwise_compatible_count(vec_a, vec_b):
    if isinstance(vec_a, csr_array) and isinstance(vec_b, csr_array):
        vec_a = sparse.csr_array(vec_a)
        vec_b = sparse.csr_array(vec_b)
        sklearn_dist = sklearn.metrics.pairwise_distances(
            vec_a, vec_b, metric=tanimoto_count_distance
        )
        assert tanimoto_count_distance(vec_a, vec_b) == sklearn_dist[0][0]
    else:
        vec_a = [vec_a]
        vec_b = [vec_b]
        sklearn_dist = sklearn.metrics.pairwise_distances(
            vec_a, vec_b, metric=tanimoto_count_distance
        )
        assert tanimoto_count_distance(vec_a[0], vec_b[0]) == sklearn_dist[0][0]


@pytest.mark.parametrize(
    "vec_a, vec_b",
    [
        (np.array([0, 1, 0, 1]), np.array([0, 1, 1, 0])),
        (csr_array([[0, 1, 0, 1]]), csr_array([[0, 1, 1, 0]])),
    ],
)
def test_sklearn_nearest_neighbors_compatible_binary(vec_a, vec_b):
    if isinstance(vec_a, csr_array) and isinstance(vec_b, csr_array):
        vec_a = sparse.csr_array(vec_a)
        vec_b = sparse.csr_array(vec_b)

        nn = sklearn.neighbors.NearestNeighbors(
            n_neighbors=1, metric=tanimoto_binary_distance
        )
        nn.fit(vec_a)
        sklearn_dist, _ = nn.kneighbors(vec_b)

        assert tanimoto_binary_distance(vec_a, vec_b) == sklearn_dist[0][0]
    else:
        vec_a = [vec_a]
        vec_b = [vec_b]

        nn = sklearn.neighbors.NearestNeighbors(
            n_neighbors=1, metric=tanimoto_binary_distance
        )
        nn.fit(vec_a)
        sklearn_dist, _ = nn.kneighbors(vec_b)

        assert tanimoto_binary_distance(vec_a[0], vec_b[0]) == sklearn_dist[0][0]


@pytest.mark.parametrize(
    "vec_a, vec_b",
    [
        (np.array([1, 2, 3, 4]), np.array([1, 2, 3, 5])),
        (csr_array([[1, 2, 3, 4]]), csr_array([[1, 2, 3, 5]])),
    ],
)
def test_sklearn_nearest_neighbors_compatible_count(vec_a, vec_b):
    if isinstance(vec_a, csr_array) and isinstance(vec_b, csr_array):
        vec_a = sparse.csr_array(vec_a)
        vec_b = sparse.csr_array(vec_b)

        nn = sklearn.neighbors.NearestNeighbors(
            n_neighbors=1, metric=tanimoto_count_distance
        )
        nn.fit(vec_a)
        sklearn_dist, _ = nn.kneighbors(vec_b)

        assert tanimoto_count_distance(vec_a, vec_b) == sklearn_dist[0][0]
    else:
        vec_a = [vec_a]
        vec_b = [vec_b]

        nn = sklearn.neighbors.NearestNeighbors(
            n_neighbors=1, metric=tanimoto_binary_distance
        )
        nn.fit(vec_a)
        sklearn_dist, _ = nn.kneighbors(vec_b)

        assert tanimoto_binary_distance(vec_a[0], vec_b[0]) == sklearn_dist[0][0]


def test_check_nan_numpy():
    vec_a = np.array([1, 2, np.nan, 4, 5])

    with pytest.raises(ValueError, match=("Input array contains NaN values")):
        _check_nan(vec_a)


def test_check_nan_scipy():
    sparse_matrix = csr_array([[1, 2, np.nan, 4, 5]])

    with pytest.raises(ValueError, match=("Input sparse matrix contains NaN values")):
        _check_nan(sparse_matrix)


def test_binary_different_types_raise_error(binary_numpy_array, binary_csr_array):
    with pytest.raises(TypeError) as exc_info:
        tanimoto_binary_similarity(binary_numpy_array, binary_csr_array)

    assert "Both vec_a and vec_b must be of the same type," in str(exc_info)
    assert (
        "got <class 'numpy.ndarray'> and <class 'scipy.sparse._csr.csr_array'>"
        in str(exc_info)
    )


def test_count_different_types_raise_error(binary_numpy_array, binary_csr_array):
    with pytest.raises(TypeError) as exc_info:
        tanimoto_count_similarity(binary_numpy_array, binary_csr_array)

    assert "Both vec_a and vec_b must be of the same type," in str(exc_info)
    assert "<class 'numpy.ndarray'> and <class 'scipy.sparse._csr.csr_array'>" in str(
        exc_info
    )


def test_check_nan_wrong_type():
    with pytest.raises(TypeError) as exc_info:
        _check_nan(1)

    assert "Expected numpy.ndarray or scipy.sparse.csr_array" in str(exc_info)
