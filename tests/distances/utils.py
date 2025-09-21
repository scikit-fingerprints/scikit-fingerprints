from collections.abc import Callable

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array

from skfp.fingerprints import ECFPFingerprint


def run_test_similarity_and_distance(
    sim_func: Callable,
    dist_func: Callable,
    vec_a: list[int],
    vec_b: list[int],
    comparison: str,
    similarity: float,
    distance: float,
    **sim_func_kwargs,
):
    vec_a = np.array(vec_a)
    vec_b = np.array(vec_b)

    vec_a_sparse = csr_array([vec_a])
    vec_b_sparse = csr_array([vec_b])

    sim_dense = sim_func(vec_a, vec_b, **sim_func_kwargs)
    dist_dense = dist_func(vec_a, vec_b)

    sim_sparse = sim_func(vec_a_sparse, vec_b_sparse, **sim_func_kwargs)
    dist_sparse = dist_func(vec_a_sparse, vec_b_sparse)

    assert_similarity_values(sim_dense, comparison, similarity)
    assert_similarity_values(sim_sparse, comparison, similarity)

    assert_distance_values(dist_dense, comparison, distance)
    assert_distance_values(dist_sparse, comparison, distance)

    assert np.isclose(sim_dense, sim_sparse)
    assert np.isclose(dist_dense, dist_sparse)


def run_test_bulk_similarity_and_distance(
    mols_list: list[Mol],
    sim_func: Callable,
    dist_func: Callable,
    bulk_sim_func: Callable,
    bulk_dist_func: Callable,
    count: bool = False,
    **sim_func_kwargs,
):
    fp = ECFPFingerprint(count=count)
    fps = fp.transform(mols_list[:10])
    fps_sparse = csr_array(fps)

    expected_sim = [
        [sim_func(fps[i], fps[j], **sim_func_kwargs) for j in range(len(fps))]
        for i in range(len(fps))
    ]
    expected_dist = [
        [dist_func(fps[i], fps[j]) for j in range(len(fps))] for i in range(len(fps))
    ]

    bulk_sim = bulk_sim_func(fps, **sim_func_kwargs)
    bulk_dist = bulk_dist_func(fps)

    bulk_sim_sparse = bulk_sim_func(fps_sparse, **sim_func_kwargs)
    bulk_dist_sparse = bulk_dist_func(fps_sparse)

    assert np.allclose(expected_sim, bulk_sim, atol=1e-3)
    assert np.allclose(expected_dist, bulk_dist, atol=1e-3)

    assert np.allclose(bulk_sim, bulk_sim_sparse, atol=1e-3)
    assert np.allclose(bulk_dist, bulk_dist_sparse, atol=1e-3)


def run_test_bulk_similarity_and_distance_two_arrays(
    mols_list: list[Mol],
    sim_func: Callable,
    dist_func: Callable,
    bulk_sim_func: Callable,
    bulk_dist_func: Callable,
    count: bool = False,
    **sim_func_kwargs,
):
    fp = ECFPFingerprint(count=count)
    fps_1 = fp.transform(mols_list[:10])
    fps_2 = fp.transform(mols_list[5:])

    fps_sparse_1 = csr_array(fps_1)
    fps_sparse_2 = csr_array(fps_2)

    expected_sim = [
        [sim_func(fps_1[i], fps_2[j], **sim_func_kwargs) for j in range(len(fps_2))]
        for i in range(len(fps_1))
    ]
    expected_dist = [
        [dist_func(fps_1[i], fps_2[j]) for j in range(len(fps_2))]
        for i in range(len(fps_1))
    ]

    bulk_sim = bulk_sim_func(fps_1, fps_2, **sim_func_kwargs)
    bulk_dist = bulk_dist_func(fps_1, fps_2)

    bulk_sim_sparse = bulk_sim_func(fps_sparse_1, fps_sparse_2, **sim_func_kwargs)
    bulk_dist_sparse = bulk_dist_func(fps_sparse_1, fps_sparse_2)

    assert np.allclose(expected_sim, bulk_sim, atol=1e-3)
    assert np.allclose(expected_dist, bulk_dist, atol=1e-3)

    assert np.allclose(bulk_sim, bulk_sim_sparse, atol=1e-3)
    assert np.allclose(bulk_dist, bulk_dist_sparse, atol=1e-3)


def assert_similarity_values(similarity: float, comparison: str, value: float) -> None:
    if comparison == ">":
        assert similarity > value
    elif comparison == "<":
        assert similarity < value
    elif comparison == "==":
        assert np.isclose(similarity, value, atol=1e-3)


def assert_distance_values(distance: float, comparison: str, value: float) -> None:
    if comparison == ">":
        assert value > distance
    elif comparison == "<":
        assert value < distance
    elif comparison == "==":
        assert np.isclose(distance, value, atol=1e-3)
