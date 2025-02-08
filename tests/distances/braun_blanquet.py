import numpy as np
import pytest
from scipy.sparse import csr_array

from skfp.distances.braun_blanquet import (
    braun_blanquet_binary_distance,
    braun_blanquet_binary_similarity,
)


@pytest.mark.parametrize(
    "vec_a, vec_b, expected_comparison",
    [
        (np.array([1, 0, 0, 0]), np.array([1, 1, 1, 1]), "<"),
        (np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1]), ">"),
        (csr_array([[1, 0, 0, 0]]), csr_array([[1, 1, 1, 1]]), "<"),
        (csr_array([[1, 1, 1, 1]]), csr_array([[1, 1, 1, 1]]), ">"),
    ],
)
def test_braun_blanquet_binary_against_threshold(vec_a, vec_b, expected_comparison):
    threshold = 0.5
    similarity = braun_blanquet_binary_similarity(vec_a, vec_b)

    if expected_comparison == ">":
        assert similarity > threshold
    elif expected_comparison == "<":
        assert similarity < threshold


@pytest.mark.parametrize(
    "vec_a, vec_b, expected_comparison",
    [
        (np.array([1, 0, 0, 0]), np.array([1, 1, 1, 1]), "<"),
        (np.array([1, 1, 1, 1]), np.array([1, 1, 1, 1]), ">"),
        (csr_array([[1, 0, 0, 0]]), csr_array([[1, 1, 1, 1]]), "<"),
        (csr_array([[1, 1, 1, 1]]), csr_array([[1, 1, 1, 1]]), ">"),
    ],
)
def test_braun_blanquet_binary_distance_against_threshold(
    vec_a, vec_b, expected_comparison
):
    threshold = 0.5
    distance = braun_blanquet_binary_distance(vec_a, vec_b)

    if expected_comparison == "<":
        assert distance > threshold
    elif expected_comparison == ">":
        assert distance < threshold
