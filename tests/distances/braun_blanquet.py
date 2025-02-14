import numpy as np
import pytest
from scipy.sparse import csr_array

from skfp.distances import (
    braun_blanquet_binary_distance,
    braun_blanquet_binary_similarity,
)
from tests.distances.utils import assert_similarity_and_distance_values


def _get_values() -> list[tuple[list[int], list[int], str, float]]:
    return [
        ([1, 0, 0], [0, 1, 1], "==", 0.0),
        ([1, 0, 0], [0, 0, 0], "==", 0.0),
        ([0, 0, 0], [0, 0, 0], "==", 1.0),
        ([1, 0, 0], [1, 0, 0], "==", 1.0),
        ([1, 1, 1], [1, 1, 1], "==", 1.0),
        ([1, 0, 0, 0], [1, 1, 1, 1], "<", 0.5),
        ([1, 1, 1, 0], [1, 1, 1, 1], ">", 0.5),
    ]


@pytest.mark.parametrize("vec_a, vec_b, comparison, value", _get_values())
def test_braun_blanquet(vec_a, vec_b, comparison, value):
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)

    vec_a_sparse = csr_array([vec_a])
    vec_b_sparse = csr_array([vec_b])

    sim_dense = braun_blanquet_binary_similarity(vec_a, vec_b)
    dist_dense = braun_blanquet_binary_distance(vec_a, vec_b)

    sim_sparse = braun_blanquet_binary_similarity(vec_a_sparse, vec_b_sparse)
    dist_sparse = braun_blanquet_binary_distance(vec_a_sparse, vec_b_sparse)

    assert_similarity_and_distance_values(sim_dense, dist_dense, comparison, value)
    assert_similarity_and_distance_values(sim_sparse, dist_sparse, comparison, value)

    assert np.isclose(sim_dense, sim_sparse)
    assert np.isclose(dist_dense, dist_sparse)
