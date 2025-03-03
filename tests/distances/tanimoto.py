import numpy as np
import pytest
from scipy.sparse import csr_array

from skfp.distances import (
    bulk_tanimoto_binary_similarity,
    bulk_tanimoto_count_similarity,
    tanimoto_binary_distance,
    tanimoto_binary_similarity,
    tanimoto_count_distance,
    tanimoto_count_similarity,
)
from skfp.distances.tanimoto import (
    bulk_tanimoto_binary_distance,
    bulk_tanimoto_count_distance,
)
from skfp.fingerprints import ECFPFingerprint
from tests.distances.utils import (
    assert_distance_values,
    assert_similarity_values,
)


def _get_binary_values() -> list[tuple[list[int], list[int], str, float, float]]:
    # vec_a, vec_b, comparison, similarity, distance
    return [
        ([1, 0, 0], [0, 1, 1], "==", 0.0, 1.0),
        ([1, 0, 0], [0, 0, 0], "==", 0.0, 1.0),
        ([0, 0, 0], [0, 0, 0], "==", 1.0, 0.0),
        ([1, 0, 0], [1, 0, 0], "==", 1.0, 0.0),
        ([1, 1, 1], [1, 1, 1], "==", 1.0, 0.0),
        ([1, 0, 0, 0], [1, 1, 1, 1], "<", 0.5, 0.5),
        ([1, 1, 1, 0], [1, 1, 1, 1], ">", 0.5, 0.5),
    ]


def _get_count_values() -> list[tuple[list[int], list[int], str, float, float]]:
    # vec_a, vec_b, comparison, similarity, distance
    return [
        ([1, 0, 0], [0, 2, 3], "==", 0.0, 1.0),
        ([1, 0, 0], [0, 0, 0], "==", 0.0, 1.0),
        ([0, 0, 0], [0, 0, 0], "==", 1.0, 0.0),
        ([1, 0, 0], [1, 0, 0], "==", 1.0, 0.0),
        ([4, 0, 0], [4, 0, 0], "==", 1.0, 0.0),
        ([1, 1, 1], [1, 1, 1], "==", 1.0, 0.0),
        ([3, 2, 1], [3, 2, 1], "==", 1.0, 0.0),
        ([3, 0, 0, 0], [1, 1, 1, 1], "<", 0.5, 0.5),
        ([2, 3, 4, 0], [2, 3, 4, 2], ">", 0.5, 0.5),
    ]


@pytest.mark.parametrize(
    "vec_a, vec_b, comparison, similarity, distance", _get_binary_values()
)
def test_tanimoto_binary(vec_a, vec_b, comparison, similarity, distance):
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)

    vec_a_sparse = csr_array([vec_a])
    vec_b_sparse = csr_array([vec_b])

    sim_dense = tanimoto_binary_similarity(vec_a, vec_b)
    dist_dense = tanimoto_binary_distance(vec_a, vec_b)

    sim_sparse = tanimoto_binary_similarity(vec_a_sparse, vec_b_sparse)
    dist_sparse = tanimoto_binary_distance(vec_a_sparse, vec_b_sparse)

    assert_similarity_values(sim_dense, comparison, similarity)
    assert_similarity_values(sim_sparse, comparison, similarity)

    assert_distance_values(dist_dense, comparison, distance)
    assert_distance_values(dist_sparse, comparison, distance)

    assert np.isclose(sim_dense, sim_sparse)
    assert np.isclose(dist_dense, dist_sparse)


@pytest.mark.parametrize(
    "vec_a, vec_b, comparison, similarity, distance", _get_count_values()
)
def test_tanimoto_count(vec_a, vec_b, comparison, similarity, distance):
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)

    vec_a_sparse = csr_array([vec_a])
    vec_b_sparse = csr_array([vec_b])

    sim_dense = tanimoto_count_similarity(vec_a, vec_b)
    dist_dense = tanimoto_count_distance(vec_a, vec_b)

    sim_sparse = tanimoto_count_similarity(vec_a_sparse, vec_b_sparse)
    dist_sparse = tanimoto_count_distance(vec_a_sparse, vec_b_sparse)

    assert_similarity_values(sim_dense, comparison, similarity)
    assert_similarity_values(sim_sparse, comparison, similarity)

    assert_distance_values(dist_dense, comparison, distance)
    assert_distance_values(dist_sparse, comparison, distance)

    assert np.isclose(sim_dense, sim_sparse)
    assert np.isclose(dist_dense, dist_sparse)


def test_bulk_tanimoto_binary(mols_list):
    mols = mols_list[:10]
    fps = np.array(ECFPFingerprint().transform(mols), dtype=np.float32)

    pairwise_sim = [
        [tanimoto_binary_similarity(fps[i], fps[j]) for j in range(len(fps))]
        for i in range(len(fps))
    ]
    pairwise_dist = [
        [tanimoto_binary_distance(fps[i], fps[j]) for j in range(len(fps))]
        for i in range(len(fps))
    ]

    assert np.allclose(pairwise_sim, bulk_tanimoto_binary_similarity(fps, fps))
    assert np.allclose(pairwise_dist, bulk_tanimoto_binary_distance(fps, fps))


def test_bulk_tanimoto_count(mols_list):
    mols = mols_list[:10]
    fps = np.array(ECFPFingerprint().transform(mols), dtype=np.float32)

    pairwise_sim = [
        [tanimoto_count_similarity(fps[i], fps[j]) for j in range(len(fps))]
        for i in range(len(fps))
    ]
    pairwise_dist = [
        [tanimoto_count_distance(fps[i], fps[j]) for j in range(len(fps))]
        for i in range(len(fps))
    ]

    assert np.allclose(pairwise_sim, bulk_tanimoto_count_similarity(fps, fps))
    assert np.allclose(pairwise_dist, bulk_tanimoto_count_distance(fps, fps))
