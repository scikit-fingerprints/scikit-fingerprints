import numpy as np
import pytest
from scipy.sparse import csr_array

from skfp.distances import (
    harris_lahey_binary_distance,
    harris_lahey_binary_similarity,
)
from tests.distances.utils import assert_distance_values, assert_similarity_values


def _get_unnormalized_values() -> list[tuple[list[int], list[int], str, float, float]]:
    # vec_a, vec_b, comparison, similarity, distance
    return [
        ([1, 0, 0], [0, 1, 1], "==", 0.0, 1.0),
        ([1, 0, 0], [0, 0, 0], "==", 1 / 3, 8 / 9),
        ([0, 0, 0], [0, 0, 0], "==", 1.0, 0.0),
        ([1, 0, 0], [1, 0, 0], "==", 3.0, 0.0),
        ([1, 1, 1], [1, 1, 1], "==", 1.0, 0.0),
        ([1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1], ">", 0.5, 0.9),
    ]


def _get_normalized_values() -> list[tuple[list[int], list[int], str, float, float]]:
    # vec_a, vec_b, comparison, similarity, distance
    return [
        ([1, 0, 0], [0, 1, 1], "==", 0.0, 1.0),
        ([1, 0, 0], [0, 0, 0], "==", 1 / 9, 8 / 9),
        ([0, 0, 0], [0, 0, 0], "==", 1.0, 0.0),
        ([1, 0, 0], [1, 0, 0], "==", 1.0, 0.0),
        ([1, 1, 1], [1, 1, 1], "==", 1.0, 0.0),
        ([1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1], "<", 0.5, 0.9),
    ]


@pytest.mark.parametrize(
    "vec_a, vec_b, comparison, similarity, distance", _get_unnormalized_values()
)
def test_harris_lahey_unnormalized(vec_a, vec_b, comparison, similarity, distance):
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)

    vec_a_sparse = csr_array([vec_a])
    vec_b_sparse = csr_array([vec_b])

    sim_dense = harris_lahey_binary_similarity(vec_a, vec_b)
    dist_dense = harris_lahey_binary_distance(vec_a, vec_b)

    sim_sparse = harris_lahey_binary_similarity(vec_a_sparse, vec_b_sparse)
    dist_sparse = harris_lahey_binary_distance(vec_a_sparse, vec_b_sparse)

    assert_similarity_values(sim_dense, comparison, similarity)
    assert_similarity_values(sim_sparse, comparison, similarity)

    assert_distance_values(dist_dense, comparison, distance)
    assert_distance_values(dist_sparse, comparison, distance)

    assert np.isclose(sim_dense, sim_sparse)
    assert np.isclose(dist_dense, dist_sparse)


@pytest.mark.parametrize(
    "vec_a, vec_b, comparison, similarity, distance", _get_normalized_values()
)
def test_harris_lahey_normalized(vec_a, vec_b, comparison, similarity, distance):
    # only similarity, since distance is always normalized
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)

    vec_a_sparse = csr_array([vec_a])
    vec_b_sparse = csr_array([vec_b])

    sim_dense = harris_lahey_binary_similarity(vec_a, vec_b, normalized=True)
    sim_sparse = harris_lahey_binary_similarity(
        vec_a_sparse, vec_b_sparse, normalized=True
    )

    assert_similarity_values(sim_dense, comparison, similarity)
    assert_similarity_values(sim_sparse, comparison, similarity)

    assert np.isclose(sim_dense, sim_sparse)
