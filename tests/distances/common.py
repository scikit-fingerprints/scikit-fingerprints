import inspect
from typing import Callable

import numpy as np
import pytest
from scipy.sparse import csr_array
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

import skfp.distances


def _get_distance_functions() -> list[Callable]:
    return [
        obj
        for name, obj in inspect.getmembers(skfp.distances)
        if inspect.isfunction(obj)
        and "distance" in name
        # omit Fraggle and MCS similarities, since they don't operate on vectors
        and "fraggle" not in name
        and "mcs" not in name
        and "bulk" not in name
    ]


@pytest.mark.parametrize("dist_func", _get_distance_functions())
def test_sklearn_pairwise_compatible_binary(dist_func):
    vec_a = np.array([[0, 1, 0, 1]])
    vec_b = np.array([[0, 1, 1, 0]])
    sklearn_dist = pairwise_distances(vec_a, vec_b, metric=dist_func)
    assert dist_func(vec_a[0], vec_b[0]) == sklearn_dist[0][0]

    vec_a = csr_array(vec_a)
    vec_b = csr_array(vec_b)
    sklearn_dist = pairwise_distances(vec_a, vec_b, metric=dist_func)
    assert dist_func(vec_a, vec_b) == sklearn_dist[0][0]


@pytest.mark.parametrize("dist_func", _get_distance_functions())
def test_sklearn_nearest_neighbors_compatible(dist_func):
    vec_a = np.array([[0, 1, 0, 1]])
    vec_b = np.array([[0, 1, 1, 0]])
    skfp_dist = dist_func(vec_a[0], vec_b[0])

    nn = NearestNeighbors(n_neighbors=1, metric=dist_func)
    nn.fit(vec_a)
    distances, _ = nn.kneighbors(vec_b)
    sklearn_dist = distances[0][0]
    assert np.isclose(skfp_dist, sklearn_dist)


@pytest.mark.parametrize("dist_func", _get_distance_functions())
def test_sklearn_nearest_neighbors_compatible_sparse_csr(dist_func):
    vec_a = csr_array([[0, 1, 0, 1]])
    vec_b = csr_array([[0, 1, 1, 0]])
    skfp_dist = dist_func(vec_a, vec_b)

    nn = NearestNeighbors(n_neighbors=1, metric=dist_func)
    nn.fit(vec_a)
    distances, _ = nn.kneighbors(vec_b)
    sklearn_dist = distances[0][0]
    assert np.isclose(skfp_dist, sklearn_dist)


@pytest.mark.parametrize("dist_func", _get_distance_functions())
def test_different_types_raise_error(dist_func):
    vec_numpy = np.array([1, 0, 1, 0, 1])
    vec_scipy = csr_array([[1, 0, 1, 0, 1]])

    with pytest.raises(TypeError) as exc_info:
        dist_func(vec_numpy, vec_scipy)

    assert "Both vec_a and vec_b must be of the same type," in str(exc_info)
    assert (
        "got <class 'numpy.ndarray'> and <class 'scipy.sparse._csr.csr_array'>"
        in str(exc_info)
    )
