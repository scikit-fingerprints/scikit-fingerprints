from typing import Optional

import numpy as np
import pytest
from scipy.sparse import csr_array

from skfp.distances import (
    bulk_tanimoto_binary_distance,
    bulk_tanimoto_binary_similarity,
    tanimoto_binary_distance,
    tanimoto_binary_similarity,
    tanimoto_count_distance,
    tanimoto_count_similarity,
)
from tests.distances.utils import (
    assert_distance_values,
    assert_matrix_distance_values,
    assert_matrix_similarity_values,
    assert_similarity_values,
)


def _get_binary_vectors() -> list[tuple[list[int], list[int], str, float, float]]:
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


def _get_count_vectors() -> list[tuple[list[int], list[int], str, float, float]]:
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


def _get_binary_matrices() -> list[
    tuple[np.ndarray, Optional[np.ndarray], str, float, float]
]:
    # X, Y, comparison, similarity, distance
    return [
        (
            [[1, 0, 1], [0, 0, 1]],
            [[1, 0, 1], [0, 0, 1]],
            "==",
            np.array([[1.0, 0.5], [0.5, 1.0]]),
            np.array([[0.0, 0.5], [0.5, 0.0]]),
        ),
        (
            [[1, 0, 1], [1, 0, 1]],
            [[0, 1, 0], [0, 1, 0]],
            "==",
            np.array([[0.0, 0.0], [0.0, 0.0]]),
            np.array([[1.0, 1.0], [1.0, 1.0]]),
        ),
        (
            [[1, 0, 1], [1, 0, 1]],
            None,
            "==",
            np.array([[1.0, 1.0], [1.0, 1.0]]),
            np.array([[0.0, 0.0], [0.0, 0.0]]),
        ),
        (
            [[1, 0, 1], [0, 1, 0]],
            None,
            "==",
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
        ),
    ]


def _get_count_matrices() -> list[tuple[np.ndarray, np.ndarray, str, float, float]]:
    # X, Y, comparison, similarity, distance
    return [
        (
            [[1, 0, 1], [0, 0, 1]],
            [[1, 0, 1], [0, 0, 1]],
            "==",
            np.array([[1.0, 0.5], [0.5, 1.0]]),
            np.array([[0.0, 0.5], [0.5, 0.0]]),
        ),
        (
            [[1, 0, 1], [1, 0, 1]],
            [[0, 1, 0], [0, 1, 0]],
            "==",
            np.array([[0.0, 0.0], [0.0, 0.0]]),
            np.array([[1.0, 1.0], [1.0, 1.0]]),
        ),
        (
            [[1, 0, 1], [1, 0, 1]],
            None,
            "==",
            np.array([[1.0, 1.0], [1.0, 1.0]]),
            np.array([[0.0, 0.0], [0.0, 0.0]]),
        ),
        (
            [[1, 0, 1], [0, 1, 0]],
            None,
            "==",
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([[0.0, 1.0], [1.0, 0.0]]),
        ),
    ]


@pytest.mark.parametrize(
    "vec_a, vec_b, comparison, similarity, distance", _get_binary_vectors()
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
    "vec_a, vec_b, comparison, similarity, distance", _get_count_vectors()
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


@pytest.mark.parametrize(
    "X, Y, comparison, similarity, distance", _get_binary_matrices()
)
def test_bulk_tanimoto_binary(X, Y, comparison, similarity, distance):
    x = np.array(X)
    y = np.array(Y) if Y is not None else None

    if y is None:
        sim_dense = bulk_tanimoto_binary_similarity(x)
        dist_dense = bulk_tanimoto_binary_distance(x)
    else:
        sim_dense = bulk_tanimoto_binary_similarity(x, y)
        dist_dense = bulk_tanimoto_binary_distance(x, y)

    assert_matrix_similarity_values(sim_dense, comparison, similarity)
    assert_matrix_distance_values(dist_dense, comparison, distance)


@pytest.mark.parametrize(
    "X, Y, comparison, similarity, distance", _get_count_matrices()
)
def test_bulk_tanimoto_count(X, Y, comparison, similarity, distance):
    x = np.array(X)
    y = np.array(Y) if Y is not None else None

    if y is None:
        sim_dense = bulk_tanimoto_binary_similarity(x)
        dist_dense = bulk_tanimoto_binary_distance(x)
    else:
        sim_dense = bulk_tanimoto_binary_similarity(x, y)
        dist_dense = bulk_tanimoto_binary_distance(x, y)

    assert_matrix_similarity_values(sim_dense, comparison, similarity)
    assert_matrix_distance_values(dist_dense, comparison, distance)
