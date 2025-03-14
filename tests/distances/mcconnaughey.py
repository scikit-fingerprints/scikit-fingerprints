import numpy as np
import pytest
from scipy.sparse import csr_array

from skfp.distances import mcconnaughey_binary_distance, mcconnaughey_binary_similarity
from skfp.distances.mcconnaughey import (
    bulk_mcconnaughey_binary_distance,
    bulk_mcconnaughey_binary_similarity,
)
from skfp.fingerprints.ecfp import ECFPFingerprint
from tests.distances.utils import (
    assert_distance_values,
    assert_similarity_values,
)


def _get_values() -> list[tuple[list[int], list[int], str, float, float, bool]]:
    # vec_a, vec_b, comparison, similarity, distance, normalized
    return [
        ([1, 0, 0], [0, 1, 1], "==", 0.0, 1.0, True),
        ([1, 0, 0], [0, 0, 0], "==", 0.0, 1.0, True),
        ([0, 0, 0], [0, 0, 0], "==", 1.0, 0.0, True),
        ([1, 0, 0], [1, 0, 0], "==", 1.0, 0.0, True),
        ([1, 1, 1], [1, 1, 1], "==", 1.0, 0.0, True),
        ([1, 1, 1, 0], [1, 1, 1, 1], ">", 0.5, 0.5, True),
        ([1, 0, 0], [0, 1, 1], "==", -1.0, 1.0, False),
        ([0, 0, 0], [0, 0, 0], "==", 1.0, 0.0, False),
        ([1, 0, 0], [1, 0, 0], "==", 1.0, 0.0, False),
        ([1, 1, 1], [1, 1, 1], "==", 1.0, 0.0, False),
    ]


@pytest.mark.parametrize(
    "vec_a, vec_b, comparison, similarity, distance, normalized", _get_values()
)
def test_mcconnaughey(vec_a, vec_b, comparison, similarity, distance, normalized):
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)

    vec_a_sparse = csr_array([vec_a])
    vec_b_sparse = csr_array([vec_b])

    sim_dense = mcconnaughey_binary_similarity(vec_a, vec_b, normalized)
    dist_dense = mcconnaughey_binary_distance(vec_a, vec_b)

    sim_sparse = mcconnaughey_binary_similarity(vec_a_sparse, vec_b_sparse, normalized)
    dist_sparse = mcconnaughey_binary_distance(vec_a_sparse, vec_b_sparse)

    assert_similarity_values(sim_dense, comparison, similarity)
    assert_similarity_values(sim_sparse, comparison, similarity)

    assert_distance_values(dist_dense, comparison, distance)
    assert_distance_values(dist_sparse, comparison, distance)

    assert np.isclose(sim_dense, sim_sparse)
    assert np.isclose(dist_dense, dist_sparse)


def test_bulk_mcconaughey_binary(mols_list):
    fp = ECFPFingerprint()
    fps = fp.transform(mols_list[:10])

    pairwise_sim = [
        [mcconnaughey_binary_similarity(fps[i], fps[j]) for j in range(len(fps))]
        for i in range(len(fps))
    ]
    pairwise_dist = [
        [mcconnaughey_binary_distance(fps[i], fps[j]) for j in range(len(fps))]
        for i in range(len(fps))
    ]

    bulk_sim = bulk_mcconnaughey_binary_similarity(fps)
    bulk_dist = bulk_mcconnaughey_binary_distance(fps)

    assert np.allclose(pairwise_sim, bulk_sim)
    assert np.allclose(pairwise_dist, bulk_dist)


def test_bulk_mcconaughey_second_array(mols_list):
    fp = ECFPFingerprint()
    fps = fp.transform(mols_list[:10])

    bulk_sim_single = bulk_mcconnaughey_binary_similarity(fps)
    bulk_sim_two = bulk_mcconnaughey_binary_similarity(fps, fps)
    assert np.allclose(bulk_sim_single, bulk_sim_two)
