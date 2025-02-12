import numpy as np
import pytest
from scipy.sparse import csr_array

from skfp.distances.rand import (
    rand_binary_distance,
    rand_binary_similarity,
)


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


@pytest.mark.parametrize(
    "vec_a, vec_b, expected_similarity",
    [
        (np.array([0, 0, 0]), np.array([0, 0, 0]), 1.0),
        (np.array([1, 0, 0]), np.array([0, 1, 1]), 0.0),
        (np.array([1, 0, 0]), np.array([0, 0, 0]), 0.0),
        (np.array([1, 0, 0]), np.array([1, 0, 0]), 1.0),
        (np.array([1, 1, 1]), np.array([1, 1, 1]), 1.0),
    ],
)
def test_rand_equality(vec_a, vec_b, expected_similarity):
    similarity = rand_binary_similarity(vec_a, vec_b)
    assert np.isclose(similarity, expected_similarity)
