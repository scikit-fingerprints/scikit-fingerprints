import numpy as np
import pytest
from scipy.sparse import csr_array

from skfp.distances.rand import rand_binary_distance, rand_binary_similarity


@pytest.mark.parametrize(
    "data_type, similarity_function, expected_similarity",
    [
        (np.zeros, rand_binary_similarity, 0.0),
        (np.ones, rand_binary_similarity, 1.0),
    ],
)
@pytest.mark.parametrize("matrix_type", ["numpy", "scipy"])
def test_rand_similarity(
    data_type, similarity_function, expected_similarity, matrix_type
):
    size = 5
    if matrix_type == "numpy":
        vec_a = data_type(size, dtype=int)
        vec_b = data_type(size, dtype=int)
    elif matrix_type == "scipy":
        vec_a = csr_array(data_type((size, size), dtype=int))
        vec_b = csr_array(data_type((size, size), dtype=int))

    assert similarity_function(vec_a, vec_b) == expected_similarity


@pytest.mark.parametrize(
    "vec_a, vec_b, expected_comparison",
    [
        (np.array([1, 1, 0, 1]), np.array([1, 1, 1, 1]), ">"),
        (np.array([1, 0, 0, 0]), np.array([1, 1, 1, 1]), "<"),
        (csr_array([[1, 1, 0, 1]]), csr_array([[1, 1, 1, 1]]), ">"),
        (csr_array([[1, 0, 0, 0]]), csr_array([[1, 1, 1, 1]]), "<"),
    ],
)
def test_rand_binary_against_threshold(vec_a, vec_b, expected_comparison):
    threshold = 0.333
    similarity = rand_binary_similarity(vec_a, vec_b)

    if expected_comparison == "==":
        assert np.isclose(similarity, threshold, atol=1e-3)
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
def test_rand_binary_distance_against_threshold(
    vec_a,
    vec_b,
    expected_comparison,
):
    threshold = 0.5
    distance = rand_binary_distance(vec_a, vec_b)

    if expected_comparison == "<":
        assert distance < threshold
    elif expected_comparison == ">":
        assert distance > threshold
