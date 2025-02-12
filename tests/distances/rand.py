import numpy as np
import pytest
from distances.utils import assert_similarity_and_distance_values
from scipy.sparse import csr_array

from skfp.distances import rand_binary_distance, rand_binary_similarity


def _get_values() -> list[tuple[list[int], list[int], str, float]]:
    return [
        ([1, 0, 0], [0, 1, 1], "==", 0.0),
        ([1, 0, 0], [0, 0, 0], "==", 0.0),
        ([0, 0, 0], [0, 0, 0], "==", 0.0),
        ([1, 0, 0], [1, 0, 0], "==", 1 / 3),
        ([1, 1, 1], [1, 1, 1], "==", 1.0),
        ([1, 0, 0, 0], [1, 1, 1, 1], "<", 0.5),
        ([1, 1, 1, 0], [1, 1, 1, 1], ">", 0.5),
    ]


@pytest.mark.parametrize("vec_a, vec_b, comparison, value", _get_values())
def test_rand(vec_a, vec_b, comparison, value):
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)

    similarity = rand_binary_similarity(vec_a, vec_b)
    distance = rand_binary_distance(vec_a, vec_b)

    assert_similarity_and_distance_values(similarity, distance, comparison, value)


@pytest.mark.parametrize("vec_a, vec_b, comparison, value", _get_values())
def test_rand_sparse(vec_a, vec_b, comparison, value):
    vec_a = csr_array([vec_a])
    vec_b = csr_array([vec_b])

    similarity = rand_binary_similarity(vec_a, vec_b)
    distance = rand_binary_distance(vec_a, vec_b)

    assert_similarity_and_distance_values(similarity, distance, comparison, value)
