import numpy as np
import pytest
from scipy import sparse
from scipy.sparse import csr_array

from skfp.distances.dice import (
    dice_binary_distance,
    dice_binary_similarity,
    dice_count_distance,
    dice_count_similarity,
)


@pytest.mark.parametrize(
    "vec_a, vec_b, expected_similarity",
    [
        (np.array([1, 2, 3, 4]), np.array([1, 2, 3, 4]), 1.0),
        (np.array([1, 2, 3, 4]), np.array([0, 0, 0, 0]), 0.0),
        (np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0]), 1.0),
        (np.array([0, 1, 0, 0]), np.array([0, 0, 2, 3]), 0.0),
    ],
)
def test_dice_count_numpy(vec_a, vec_b, expected_similarity):
    assert dice_count_similarity(vec_a, vec_b) == expected_similarity


@pytest.mark.parametrize(
    "data_type, similarity_function",
    [
        (np.zeros, dice_binary_similarity),
        (np.ones, dice_binary_similarity),
        (np.zeros, dice_count_similarity),
        (np.ones, dice_count_similarity),
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
def test_dice_binary_against_threshold(vec_a, vec_b, expected_comparison):
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
        (np.array([1, 0, 3, 0]), np.array([1, 0, 3, 0]), ">"),
        (np.array([1, 7, 3, 9]), np.array([1, 1, 6, 0]), "<"),
        (csr_array([[1, 1, 0, 1]]), csr_array([[1, 1, 0, 1]]), "=="),
        (csr_array([[1, 0, 3, 0]]), csr_array([[1, 0, 3, 0]]), ">"),
        (csr_array([[1, 7, 3, 9]]), csr_array([[1, 1, 6, 0]]), "<"),
    ],
)
def test_dice_count_against_threshold(vec_a, vec_b, expected_comparison):
    threshold = 0.5
    similarity = dice_count_similarity(vec_a, vec_b)

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
def test_dice_binary_distance_against_threshold(
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
def test_dice_count_distance_against_threshold(vec_a, vec_b, expected_comparison):
    threshold = 0.5
    distance = dice_count_distance(vec_a, vec_b)

    if expected_comparison == "<":
        assert distance < threshold
    elif expected_comparison == ">":
        assert distance > threshold
