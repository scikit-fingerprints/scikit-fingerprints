import numpy as np
import pytest
from scipy.sparse import csr_array

from skfp.distances.simpson import (
    simpson_binary_distance,
    simpson_binary_similarity,
)


@pytest.mark.parametrize(
    "vec_a, vec_b",
    [
        (np.array([1, 0, 0, 0]), np.array([1, 1, 1, 1])),
        (np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1])),
        (csr_array([[1, 0, 0, 0]]), csr_array([[1, 1, 1, 1]])),
        (csr_array([[1, 1, 1, 1]]), csr_array([[1, 1, 1, 1]])),
    ],
)
def test_simpson_binary_against_threshold(vec_a, vec_b):
    threshold = 1
    similarity = simpson_binary_similarity(vec_a, vec_b)

    assert np.isclose(similarity, threshold, atol=1e-3)


@pytest.mark.parametrize(
    "vec_a, vec_b",
    [
        (np.array([1, 0, 0, 0]), np.array([1, 1, 1, 1])),
        (csr_array([[1, 0, 0, 0]]), csr_array([[1, 1, 1, 1]])),
    ],
)
def test_simpson_binary_distance_against_threshold(vec_a, vec_b):
    threshold = 0
    distance = simpson_binary_distance(vec_a, vec_b)

    assert np.isclose(distance, threshold, atol=1e-3)


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
def test_simpson_equality(vec_a, vec_b, expected_similarity):
    similarity = simpson_binary_distance(vec_a, vec_b)
    assert np.isclose(similarity, expected_similarity)
