import numpy as np
import pytest
from scipy.sparse import csr_array

from skfp.distances.tanimoto import (
    tanimoto_binary_distance,
    tanimoto_binary_similarity,
    tanimoto_count_distance,
    tanimoto_count_similarity,
)


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
def test_tanimoto_binary_against_threshold(vec_a, vec_b, expected_comparison):
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
def test_tanimoto_binary_distance_against_threshold(
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
def test_tanimoto_count_distance_against_threshold(vec_a, vec_b, expected_comparison):
    threshold = 0.5
    distance = tanimoto_count_distance(vec_a, vec_b)

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
def test_tanimoto_binary_equality(vec_a, vec_b, expected_similarity):
    similarity = tanimoto_binary_similarity(vec_a, vec_b)
    assert np.isclose(similarity, expected_similarity)


@pytest.mark.parametrize(
    "vec_a, vec_b, expected_similarity",
    [
        (np.array([0, 0, 0]), np.array([0, 0, 0]), 1.0),
        (np.array([1, 0, 0]), np.array([0, 2, 3]), 0.0),
        (np.array([2, 0, 0]), np.array([0, 0, 0]), 0.0),
        (np.array([4, 0, 0]), np.array([4, 0, 0]), 1.0),
        (np.array([5, 4, 3]), np.array([5, 4, 3]), 1.0),
    ],
)
def test_tanimoto_count_equality(vec_a, vec_b, expected_similarity):
    similarity = tanimoto_binary_similarity(vec_a, vec_b)
    assert np.isclose(similarity, expected_similarity)
