import numpy as np
import pytest
from distances.utils import assert_similarity_and_distance_values
from scipy.sparse import csr_array

from skfp.distances import (
    dice_binary_distance,
    dice_binary_similarity,
    dice_count_distance,
    dice_count_similarity,
)


def _get_binary_values() -> list[tuple[list[int], list[int], str, float]]:
    return [
        ([1, 0, 0], [0, 1, 1], "==", 0.0),
        ([1, 0, 0], [0, 0, 0], "==", 0.0),
        ([0, 0, 0], [0, 0, 0], "==", 1.0),
        ([1, 0, 0], [1, 0, 0], "==", 1.0),
        ([1, 1, 1], [1, 1, 1], "==", 1.0),
        ([1, 0, 0, 0], [1, 1, 1, 1], "<", 0.5),
        ([1, 1, 1, 0], [1, 1, 1, 1], ">", 0.5),
    ]


def _get_count_values() -> list[tuple[list[int], list[int], str, float]]:
    return [
        ([1, 0, 0], [0, 2, 3], "==", 0.0),
        ([1, 0, 0], [0, 0, 0], "==", 0.0),
        ([0, 0, 0], [0, 0, 0], "==", 1.0),
        ([1, 0, 0], [1, 0, 0], "==", 1.0),
        ([4, 0, 0], [4, 0, 0], "==", 1.0),
        ([1, 1, 1], [1, 1, 1], "==", 1.0),
        ([3, 2, 1], [3, 2, 1], "==", 1.0),
        ([3, 0, 0, 0], [1, 1, 1, 1], "<", 0.5),
        ([2, 3, 4, 0], [2, 3, 4, 2], ">", 0.5),
    ]


@pytest.mark.parametrize("vec_a, vec_b, comparison, value", _get_binary_values())
def test_dice_binary(vec_a, vec_b, comparison, value):
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)

    similarity = dice_binary_similarity(vec_a, vec_b)
    distance = dice_binary_distance(vec_a, vec_b)

    assert_similarity_and_distance_values(similarity, distance, comparison, value)


@pytest.mark.parametrize("vec_a, vec_b, comparison, value", _get_binary_values())
def test_dice_binary_sparse(vec_a, vec_b, comparison, value):
    vec_a = csr_array([vec_a])
    vec_b = csr_array([vec_b])

    similarity = dice_binary_similarity(vec_a, vec_b)
    distance = dice_binary_distance(vec_a, vec_b)

    assert_similarity_and_distance_values(similarity, distance, comparison, value)


@pytest.mark.parametrize("vec_a, vec_b, comparison, value", _get_count_values())
def test_dice_count(vec_a, vec_b, comparison, value):
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)

    similarity = dice_count_similarity(vec_a, vec_b)
    distance = dice_count_distance(vec_a, vec_b)

    assert_similarity_and_distance_values(similarity, distance, comparison, value)


@pytest.mark.parametrize("vec_a, vec_b, comparison, value", _get_count_values())
def test_dice_count_sparse(vec_a, vec_b, comparison, value):
    vec_a = csr_array([vec_a])
    vec_b = csr_array([vec_b])

    similarity = dice_count_similarity(vec_a, vec_b)
    distance = dice_count_distance(vec_a, vec_b)

    assert_similarity_and_distance_values(similarity, distance, comparison, value)
