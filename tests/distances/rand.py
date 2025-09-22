import pytest

from skfp.distances import rand_binary_distance, rand_binary_similarity
from skfp.distances.rand import bulk_rand_binary_distance, bulk_rand_binary_similarity
from tests.distances.utils import (
    run_test_bulk_similarity_and_distance,
    run_test_bulk_similarity_and_distance_two_arrays,
    run_test_similarity_and_distance,
)


def _get_values() -> list[tuple[list[int], list[int], str, float, float]]:
    # vec_a, vec_b, comparison, similarity, distance
    return [
        ([1, 0, 0], [0, 1, 1], "==", 0.0, 1.0),
        ([1, 0, 0], [0, 0, 0], "==", 2 / 3, 1 / 3),
        ([0, 0, 0], [0, 0, 0], "==", 1.0, 0.0),
        ([1, 0, 0], [1, 0, 0], "==", 1.0, 0.0),
        ([1, 1, 1], [1, 1, 1], "==", 1.0, 0.0),
        ([1, 0, 0, 0], [1, 1, 1, 1], "<", 0.5, 0.5),
        ([1, 1, 1, 0], [1, 1, 1, 1], ">", 0.5, 0.5),
    ]


@pytest.mark.parametrize(
    "vec_a, vec_b, comparison, similarity, distance", _get_values()
)
def test_rand(vec_a, vec_b, comparison, similarity, distance):
    run_test_similarity_and_distance(
        rand_binary_similarity,
        rand_binary_distance,
        vec_a,
        vec_b,
        comparison,
        similarity,
        distance,
    )


def test_bulk_rand(mols_list):
    run_test_bulk_similarity_and_distance(
        mols_list,
        rand_binary_similarity,
        rand_binary_distance,
        bulk_rand_binary_similarity,
        bulk_rand_binary_distance,
    )


def test_bulk_rand_second_array(mols_list):
    run_test_bulk_similarity_and_distance_two_arrays(
        mols_list,
        rand_binary_similarity,
        rand_binary_distance,
        bulk_rand_binary_similarity,
        bulk_rand_binary_distance,
    )
