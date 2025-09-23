import numpy as np
import pytest
from scipy.sparse import csr_array

from skfp.distances import russell_binary_distance, russell_binary_similarity
from skfp.distances.russell import (
    bulk_russell_binary_distance,
    bulk_russell_binary_similarity,
)
from skfp.fingerprints.ecfp import ECFPFingerprint


def _get_values() -> list[tuple[list[int], list[int], float, float]]:
    # vec_a, vec_b, similarity, distance
    return [
        ([1, 0, 0], [0, 1, 1], 0.0, 1.0),
        ([1, 0, 0], [0, 0, 0], 0.0, 1.0),
        ([0, 0, 0], [0, 0, 0], 0.0, 1.0),
        ([1, 1, 1], [1, 1, 1], 1.0, 0.0),
        ([1, 0, 0], [1, 0, 0], 1 / 3, 2 / 3),
        ([1, 1, 1], [1, 1, 1], 1.0, 0.0),
        ([1, 1, 1, 1], [1, 1, 0, 0], 0.5, 0.5),
    ]


@pytest.mark.parametrize("vec_a, vec_b, similarity, distance", _get_values())
def test_russell(vec_a, vec_b, similarity, distance):
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)

    vec_a_sparse = csr_array([vec_a])
    vec_b_sparse = csr_array([vec_b])

    sim_dense = russell_binary_similarity(vec_a, vec_b)
    dist_dense = russell_binary_distance(vec_a, vec_b)

    sim_sparse = russell_binary_similarity(vec_a_sparse, vec_b_sparse)
    dist_sparse = russell_binary_distance(vec_a_sparse, vec_b_sparse)

    assert np.isclose(sim_dense, similarity, atol=1e-3)
    assert np.isclose(sim_sparse, similarity, atol=1e-3)

    assert np.isclose(dist_dense, distance, atol=1e-3)
    assert np.isclose(dist_sparse, distance, atol=1e-3)

    assert np.isclose(sim_dense, sim_sparse)
    assert np.isclose(dist_dense, dist_sparse)


def test_bulk_russell_binary(mols_list):
    fp = ECFPFingerprint()
    fps = fp.transform(mols_list[:10])

    pairwise_sim = [
        [russell_binary_similarity(fps[i], fps[j]) for j in range(len(fps))]
        for i in range(len(fps))
    ]
    pairwise_dist = [
        [russell_binary_distance(fps[i], fps[j]) for j in range(len(fps))]
        for i in range(len(fps))
    ]

    bulk_sim = bulk_russell_binary_similarity(fps)
    bulk_dist = bulk_russell_binary_distance(fps)

    assert np.allclose(pairwise_sim, bulk_sim)
    assert np.allclose(pairwise_dist, bulk_dist)


def test_bulk_russell_second_array(mols_list):
    fp = ECFPFingerprint()
    fps = fp.transform(mols_list[:10])

    bulk_sim_single = bulk_russell_binary_similarity(fps)
    bulk_sim_two = bulk_russell_binary_similarity(fps, fps)
    assert np.allclose(bulk_sim_single, bulk_sim_two)
