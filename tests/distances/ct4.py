import numpy as np
import pytest
from scipy.sparse import csr_array

from skfp.distances import (
    ct4_binary_distance,
    ct4_binary_similarity,
    ct4_count_distance,
    ct4_count_similarity,
)
from tests.distances.utils import assert_similarity_and_distance_values


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
        ([4, 0, 0, 0], [1, 2, 2, 2], "<", 0.5),
        ([2, 3, 4, 0], [2, 3, 4, 2], ">", 0.5),
    ]


@pytest.mark.parametrize("vec_a, vec_b, comparison, value", _get_binary_values())
def test_ct4_binary(vec_a, vec_b, comparison, value):
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)

    vec_a_sparse = csr_array([vec_a])
    vec_b_sparse = csr_array([vec_b])

    sim_dense = ct4_binary_similarity(vec_a, vec_b)
    dist_dense = ct4_binary_distance(vec_a, vec_b)

    sim_sparse = ct4_binary_similarity(vec_a_sparse, vec_b_sparse)
    dist_sparse = ct4_binary_distance(vec_a_sparse, vec_b_sparse)

    assert_similarity_and_distance_values(sim_dense, dist_dense, comparison, value)
    assert_similarity_and_distance_values(sim_sparse, dist_sparse, comparison, value)

    assert np.isclose(sim_dense, sim_sparse)
    assert np.isclose(dist_dense, dist_sparse)


@pytest.mark.parametrize("vec_a, vec_b, comparison, value", _get_count_values())
def test_ct4_count(vec_a, vec_b, comparison, value):
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)

    vec_a_sparse = csr_array([vec_a])
    vec_b_sparse = csr_array([vec_b])

    sim_dense = ct4_count_similarity(vec_a, vec_b)
    dist_dense = ct4_count_distance(vec_a, vec_b)

    sim_sparse = ct4_count_similarity(vec_a_sparse, vec_b_sparse)
    dist_sparse = ct4_count_distance(vec_a_sparse, vec_b_sparse)

    assert_similarity_and_distance_values(sim_dense, dist_dense, comparison, value)
    assert_similarity_and_distance_values(sim_sparse, dist_sparse, comparison, value)

    assert np.isclose(sim_dense, sim_sparse)
    assert np.isclose(dist_dense, dist_sparse)
