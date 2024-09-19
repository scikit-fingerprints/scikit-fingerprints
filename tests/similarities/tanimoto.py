import numpy as np
import pytest
import sklearn
import sklearn.metrics
import sklearn.neighbors
from scipy import sparse
from scipy.sparse import csr_matrix

from skfp.similarities.tanimoto import (
    _check_nan,
    _check_zero_denominator,
    binary_tanimoto_distance,
    binary_tanimoto_similarity,
    count_tanimoto_distance,
    count_tanimoto_similarity,
)


@pytest.fixture
def binary_numpy_arrays():
    size: int = 5
    random_binary_array = np.random.randint(0, 2, size=size)

    A = random_binary_array
    B = random_binary_array.copy()

    return A, B


@pytest.fixture
def threshold() -> float:
    return 0.5


def test_check_nan_numpy():
    A = np.array([1.0, 2.0, np.nan, 4.0, 5.0])

    with pytest.raises(ValueError, match=("Input array contains NaN values")):
        _check_nan(A)


def test_check_nan_scipy():
    dense_array = np.array([1, 2, np.nan, 4, 5])
    sparse_matrix = csr_matrix(dense_array)

    with pytest.raises(ValueError, match=("Input sparse matrix contains NaN values")):
        _check_nan(sparse_matrix)


def test_zero_denominator():
    with pytest.raises(ZeroDivisionError, match="Denominator is zero"):
        _check_zero_denominator(0)


def test_binary_wrong_types_raise_error():
    A: int = 1
    B: float = 1.0

    with pytest.raises(
        TypeError,
        match=(
            "Both A and B must be of the same type: either numpy.ndarray or scipy.sparse.csr_matrix, "
            "got <class 'int'> and <class 'float'>"
        ),
    ):
        binary_tanimoto_similarity(A, B)


def test_binary_different_types_raise_error(binary_numpy_arrays):
    A_np, _ = binary_numpy_arrays
    A_sp = sparse.csr_matrix(A_np)

    with pytest.raises(
        TypeError,
        match=(
            "Both A and B must be of the same type: either numpy.ndarray or scipy.sparse.csr_matrix, "
            "got <class 'numpy.ndarray'> and <class 'scipy.sparse._csr.csr_matrix'>"
        ),
    ):
        binary_tanimoto_similarity(A_np, A_sp)


def test_count_wrong_types_raise_error():
    A: int = 1
    B: float = 1.0

    with pytest.raises(
        TypeError,
        match=(
            "Both A and B must be of the same type: either numpy.ndarray or scipy.sparse.csr_matrix, "
            "got <class 'int'> and <class 'float'>"
        ),
    ):
        count_tanimoto_similarity(A, B)


def test_count_different_types_raise_error(binary_numpy_arrays):
    A_np, _ = binary_numpy_arrays
    A_sp = sparse.csr_matrix(A_np)

    with pytest.raises(
        TypeError,
        match=(
            "Both A and B must be of the same type: either numpy.ndarray or scipy.sparse.csr_matrix, got "
            "<class 'numpy.ndarray'> and <class 'scipy.sparse._csr.csr_matrix'>"
        ),
    ):
        count_tanimoto_similarity(A_np, A_sp)


@pytest.mark.parametrize(
    "data_type, similarity_function",
    [
        (np.zeros, binary_tanimoto_similarity),
        (np.ones, binary_tanimoto_similarity),
        (np.zeros, count_tanimoto_similarity),
        (np.ones, count_tanimoto_similarity),
    ],
)
@pytest.mark.parametrize("matrix_type", ["numpy", "scipy"])
def test_tanimoto_similarity(data_type, similarity_function, matrix_type):
    size = 5
    if matrix_type == "numpy":
        A = data_type(
            size,
            dtype=int if similarity_function == binary_tanimoto_similarity else float,
        )
        B = data_type(
            size,
            dtype=int if similarity_function == binary_tanimoto_similarity else float,
        )
    elif matrix_type == "scipy":
        A = sparse.csr_matrix(
            data_type(
                (size, size),
                dtype=(
                    int if similarity_function == binary_tanimoto_similarity else float
                ),
            )
        )
        B = sparse.csr_matrix(
            data_type(
                (size, size),
                dtype=(
                    int if similarity_function == binary_tanimoto_similarity else float
                ),
            )
        )

    assert similarity_function(A, B) == 1.0


@pytest.mark.parametrize(
    "A, B, expected_comparison",
    [
        (np.array([1, 1, 0, 1]), np.array([1, 0, 1, 1]), "=="),
        (np.array([1, 1, 0, 1]), np.array([1, 1, 1, 1]), ">"),
        (np.array([1, 0, 0, 0]), np.array([1, 1, 1, 1]), "<"),
    ],
)
@pytest.mark.parametrize("use_sparse", [False, True])
def test_binary_against_threshold(A, B, expected_comparison, threshold, use_sparse):
    if use_sparse:
        A = sparse.csr_matrix(A)
        B = sparse.csr_matrix(B)

    similarity = binary_tanimoto_similarity(A, B)

    if expected_comparison == "==":
        assert similarity == threshold
    elif expected_comparison == ">":
        assert similarity > threshold
    elif expected_comparison == "<":
        assert similarity < threshold


@pytest.mark.parametrize(
    "A, B, expected_comparison",
    [
        (np.array([1.1, 1.2, 1.3, 1.4]), np.array([1.1, 1.2, 1.3, 0.4]), ">"),
        (np.array([1.1, 1.2, 1.3, 1.4]), np.array([0.1, 0.2, 0.3, 0.4]), "<"),
    ],
)
@pytest.mark.parametrize("use_sparse", [False, True])
def test_count_against_threshold(A, B, expected_comparison, threshold, use_sparse):
    if use_sparse:
        A = sparse.csr_matrix(A)
        B = sparse.csr_matrix(B)

    similarity = binary_tanimoto_similarity(A, B)

    if expected_comparison == ">":
        assert similarity > threshold
    elif expected_comparison == "<":
        assert similarity < threshold


@pytest.mark.parametrize(
    "A, B, expected_comparison",
    [
        (np.array([1, 1, 0, 1]), np.array([1, 1, 1, 1]), "<"),
        (np.array([1, 0, 0, 0]), np.array([1, 1, 1, 1]), ">"),
    ],
)
@pytest.mark.parametrize("use_sparse", [False, True])
def test_binary_distance(A, B, expected_comparison, use_sparse, threshold):
    if use_sparse:
        A = sparse.csr_matrix(A)
        B = sparse.csr_matrix(B)

    distance = binary_tanimoto_distance(A, B)

    if expected_comparison == "<":
        assert distance < threshold
    elif expected_comparison == ">":
        assert distance > threshold


@pytest.mark.parametrize(
    "A, B, expected_comparison",
    [
        (np.array([1.1, 1.2, 1.3, 1.4]), np.array([1.1, 1.2, 1.3, 0.4]), "<"),
        (np.array([1.1, 1.2, 1.3, 1.4]), np.array([0.1, 0.2, 0.3, 0.4]), ">"),
    ],
)
@pytest.mark.parametrize("use_sparse", [False, True])
def test_count_distance(A, B, expected_comparison, use_sparse, threshold):
    if use_sparse:
        A = sparse.csr_matrix(A)
        B = sparse.csr_matrix(B)

    distance = binary_tanimoto_distance(A, B)

    if expected_comparison == "<":
        assert distance < threshold
    elif expected_comparison == ">":
        assert distance > threshold


@pytest.mark.parametrize(
    "A, B",
    [
        (np.array([0, 1, 0, 1]), np.array([0, 1, 1, 0])),
        (csr_matrix([0, 1, 0, 1]), csr_matrix([0, 1, 1, 0])),
    ],
)
def test_binary_against_sklearn_pairwise(A, B):
    if isinstance(A, csr_matrix) and isinstance(B, csr_matrix):
        sklearn_dist = sklearn.metrics.pairwise_distances(
            A, B, metric=binary_tanimoto_distance
        )
        assert binary_tanimoto_distance(A, B) == sklearn_dist[0][0]
    else:
        A = [A]
        B = [B]
        sklearn_dist = sklearn.metrics.pairwise_distances(
            A, B, metric=binary_tanimoto_distance
        )
        assert binary_tanimoto_distance(A[0], B[0]) == sklearn_dist[0][0]


@pytest.mark.parametrize(
    "A, B",
    [
        (np.array([1.1, 1.2, 1.3, 1.4]), np.array([1.1, 1.2, 1.3, 0.4])),
        (csr_matrix([1.1, 1.2, 1.3, 1.4]), csr_matrix([1.1, 1.2, 1.3, 0.4])),
    ],
)
def test_count_against_sklearn_pairwise(A, B):
    if isinstance(A, csr_matrix) and isinstance(B, csr_matrix):
        A = sparse.csr_matrix(A)
        B = sparse.csr_matrix(B)
        sklearn_dist = sklearn.metrics.pairwise_distances(
            A, B, metric=count_tanimoto_distance
        )
        assert count_tanimoto_distance(A, B) == sklearn_dist[0][0]
    else:
        A = [A]
        B = [B]
        sklearn_dist = sklearn.metrics.pairwise_distances(
            A, B, metric=count_tanimoto_distance
        )
        assert count_tanimoto_distance(A[0], B[0]) == sklearn_dist[0][0]


@pytest.mark.parametrize(
    "A, B",
    [
        (np.array([0, 1, 0, 1]), np.array([0, 1, 1, 0])),
        (csr_matrix([0, 1, 0, 1]), csr_matrix([0, 1, 1, 0])),
    ],
)
def test_binary_against_sklearn_nn(A, B):
    if isinstance(A, csr_matrix) and isinstance(B, csr_matrix):
        A = sparse.csr_matrix(A)
        B = sparse.csr_matrix(B)

        nn = sklearn.neighbors.NearestNeighbors(
            n_neighbors=1, metric=binary_tanimoto_distance
        )
        nn.fit(A)
        sklearn_dist, _ = nn.kneighbors(B)

        assert binary_tanimoto_distance(A, B) == sklearn_dist[0][0]
    else:
        A = [A]
        B = [B]

        nn = sklearn.neighbors.NearestNeighbors(
            n_neighbors=1, metric=binary_tanimoto_distance
        )
        nn.fit(A)
        sklearn_dist, _ = nn.kneighbors(B)

        assert binary_tanimoto_distance(A[0], B[0]) == sklearn_dist[0][0]


@pytest.mark.parametrize(
    "A, B",
    [
        (np.array([1.1, 1.2, 1.3, 1.4]), np.array([1.1, 1.2, 1.3, 0.4])),
        (csr_matrix([1.1, 1.2, 1.3, 1.4]), csr_matrix([1.1, 1.2, 1.3, 0.4])),
    ],
)
def test_count_against_sklearn_nn(A, B):
    if isinstance(A, csr_matrix) and isinstance(B, csr_matrix):
        A = sparse.csr_matrix(A)
        B = sparse.csr_matrix(B)

        nn = sklearn.neighbors.NearestNeighbors(
            n_neighbors=1, metric=binary_tanimoto_distance
        )
        nn.fit(A)
        sklearn_dist, _ = nn.kneighbors(B)

        assert binary_tanimoto_distance(A, B) == sklearn_dist[0][0]
    else:
        A = [A]
        B = [B]

        nn = sklearn.neighbors.NearestNeighbors(
            n_neighbors=1, metric=binary_tanimoto_distance
        )
        nn.fit(A)
        sklearn_dist, _ = nn.kneighbors(B)

        assert binary_tanimoto_distance(A[0], B[0]) == sklearn_dist[0][0]
