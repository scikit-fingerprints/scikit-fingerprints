import numpy as np
import pytest
from scipy.sparse import csr_array

from skfp.distances import (
    ct4_binary_distance,
    ct4_binary_similarity,
    ct4_count_distance,
    ct4_count_similarity,
)
from skfp.distances.ct4 import (
    bulk_ct4_binary_distance,
    bulk_ct4_binary_similarity,
    bulk_ct4_count_distance,
    bulk_ct4_count_similarity,
)
from skfp.fingerprints.ecfp import ECFPFingerprint
from tests.distances.utils import assert_distance_values, assert_similarity_values


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
        ([4, 0, 0, 0], [1, 2, 2, 2], "<", 0.5, 0.5),
        ([2, 3, 4, 0], [2, 3, 4, 2], ">", 0.5, 0.5),
    ]


@pytest.mark.parametrize(
    "vec_a, vec_b, comparison, similarity, distance", _get_binary_values()
)
def test_ct4_binary(vec_a, vec_b, comparison, similarity, distance):
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)

    vec_a_sparse = csr_array([vec_a])
    vec_b_sparse = csr_array([vec_b])

    sim_dense = ct4_binary_similarity(vec_a, vec_b)
    dist_dense = ct4_binary_distance(vec_a, vec_b)

    sim_sparse = ct4_binary_similarity(vec_a_sparse, vec_b_sparse)
    dist_sparse = ct4_binary_distance(vec_a_sparse, vec_b_sparse)

    assert_similarity_values(sim_dense, comparison, similarity)
    assert_similarity_values(sim_sparse, comparison, similarity)

    assert_distance_values(dist_dense, comparison, distance)
    assert_distance_values(dist_sparse, comparison, distance)

    assert np.isclose(sim_dense, sim_sparse)
    assert np.isclose(dist_dense, dist_sparse)


@pytest.mark.parametrize(
    "vec_a, vec_b, comparison, similarity, distance", _get_count_values()
)
def test_ct4_count(vec_a, vec_b, comparison, similarity, distance):
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)

    vec_a_sparse = csr_array([vec_a])
    vec_b_sparse = csr_array([vec_b])

    sim_dense = ct4_count_similarity(vec_a, vec_b)
    dist_dense = ct4_count_distance(vec_a, vec_b)

    sim_sparse = ct4_count_similarity(vec_a_sparse, vec_b_sparse)
    dist_sparse = ct4_count_distance(vec_a_sparse, vec_b_sparse)

    assert_similarity_values(sim_dense, comparison, similarity)
    assert_distance_values(dist_sparse, comparison, distance)

    assert np.isclose(sim_dense, sim_sparse)
    assert np.isclose(dist_dense, dist_sparse)


def test_bulk_ct4_binary(mols_list):
    fp = ECFPFingerprint()
    fps = fp.transform(mols_list[:10])

    pairwise_sim = [
        [ct4_binary_similarity(fps[i], fps[j]) for j in range(len(fps))]
        for i in range(len(fps))
    ]
    pairwise_dist = [
        [ct4_binary_distance(fps[i], fps[j]) for j in range(len(fps))]
        for i in range(len(fps))
    ]

    bulk_sim = bulk_ct4_binary_similarity(fps)
    bulk_dist = bulk_ct4_binary_distance(fps)

    assert np.allclose(pairwise_sim, bulk_sim)
    assert np.allclose(pairwise_dist, bulk_dist)


def test_bulk_ct4_count(mols_list):
    fp = ECFPFingerprint(count=True)
    fps = fp.transform(mols_list[:10])

    pairwise_sim = [
        [ct4_count_similarity(fps[i], fps[j]) for j in range(len(fps))]
        for i in range(len(fps))
    ]
    pairwise_dist = [
        [ct4_count_distance(fps[i], fps[j]) for j in range(len(fps))]
        for i in range(len(fps))
    ]

    bulk_sim = bulk_ct4_count_similarity(fps)
    bulk_dist = bulk_ct4_count_distance(fps)

    assert np.allclose(pairwise_sim, bulk_sim)
    assert np.allclose(pairwise_dist, bulk_dist)


def test_bulk_ct4_second_array(mols_list):
    fp = ECFPFingerprint()
    fps = fp.transform(mols_list[:10])

    bulk_sim_single = bulk_ct4_binary_similarity(fps)
    bulk_sim_two = bulk_ct4_binary_similarity(fps, fps)
    assert np.allclose(bulk_sim_single, bulk_sim_two)

    fp = ECFPFingerprint(count=True)
    fps = fp.transform(mols_list[:10])

    bulk_sim_single = bulk_ct4_count_similarity(fps)
    bulk_sim_two = bulk_ct4_count_similarity(fps, fps)
    assert np.allclose(bulk_sim_single, bulk_sim_two)
