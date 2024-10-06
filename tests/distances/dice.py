import numpy as np
import pytest
import sklearn
import sklearn.metrics
import sklearn.neighbors
from scipy import sparse
from scipy.sparse import csr_array

from skfp.distances.dice import dice_binary_distance, dice_binary_similarity


@pytest.mark.parametrize(
    "data_type, similarity_function",
    [
        (np.zeros, dice_binary_similarity),
        (np.ones, dice_binary_similarity),
    ],
)
@pytest.mark.parametrize("matrix_type", ["numpy", "scipy"])
def test_dice_similarity(data_type, similarity_function, matrix_type):
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
        (np.array([1, 1, 0, 1]), np.array([1, 0, 1, 1]), ">"),
        (np.array([1, 1, 0, 1]), np.array([1, 1, 1, 1]), ">"),
        (np.array([1, 0, 0, 0]), np.array([1, 1, 1, 1]), "<"),
        (csr_array([[1, 1, 0, 1]]), csr_array([[1, 0, 1, 1]]), ">"),
        (csr_array([[1, 1, 0, 1]]), csr_array([[1, 1, 1, 1]]), ">"),
        (csr_array([[1, 0, 0, 0]]), csr_array([[1, 1, 1, 1]]), "<"),
    ],
)
def test_binary_dice_against_threshold(vec_a, vec_b, expected_comparison):
    threshold = 0.5
    similarity = dice_binary_similarity(vec_a, vec_b)

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
    distance = dice_binary_distance(vec_a, vec_b)

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
            vec_a, vec_b, metric=dice_binary_distance
        )
        assert dice_binary_distance(vec_a, vec_b) == sklearn_dist[0][0]
    else:
        vec_a = [vec_a]
        vec_b = [vec_b]
        sklearn_dist = sklearn.metrics.pairwise_distances(
            vec_a, vec_b, metric=dice_binary_distance
        )
        assert dice_binary_distance(vec_a[0], vec_b[0]) == sklearn_dist[0][0]


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
            n_neighbors=1, metric=dice_binary_distance
        )
        nn.fit(vec_a)
        sklearn_dist, _ = nn.kneighbors(vec_b)

        assert dice_binary_distance(vec_a, vec_b) == sklearn_dist[0][0]
    else:
        vec_a = [vec_a]
        vec_b = [vec_b]

        nn = sklearn.neighbors.NearestNeighbors(
            n_neighbors=1, metric=dice_binary_distance
        )
        nn.fit(vec_a)
        sklearn_dist, _ = nn.kneighbors(vec_b)

        assert dice_binary_distance(vec_a[0], vec_b[0]) == sklearn_dist[0][0]


def test_binary_different_types_raise_error(binary_numpy_array, binary_csr_array):
    with pytest.raises(TypeError) as exc_info:
        dice_binary_similarity(binary_numpy_array, binary_csr_array)

    assert "Both vec_a and vec_b must be of the same type," in str(exc_info)
    assert (
        "got <class 'numpy.ndarray'> and <class 'scipy.sparse._csr.csr_array'>"
        in str(exc_info)
    )
