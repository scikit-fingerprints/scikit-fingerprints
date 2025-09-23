import pytest

from skfp.distances import russell_binary_distance, russell_binary_similarity
from skfp.distances.russell import (
    bulk_russell_binary_distance,
    bulk_russell_binary_similarity,
)
from tests.distances.utils import (
    run_test_bulk_similarity_and_distance,
    run_test_bulk_similarity_and_distance_two_arrays,
    run_test_similarity_and_distance,
)


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
    run_test_similarity_and_distance(
        russell_binary_similarity,
        russell_binary_distance,
        vec_a,
        vec_b,
        similarity,
        distance,
    )


def test_bulk_russell(mols_list):
    run_test_bulk_similarity_and_distance(
        mols_list,
        russell_binary_similarity,
        russell_binary_distance,
        bulk_russell_binary_similarity,
        bulk_russell_binary_distance,
    )


def test_bulk_russell_second_array(mols_list):
    run_test_bulk_similarity_and_distance_two_arrays(
        mols_list,
        russell_binary_similarity,
        russell_binary_distance,
        bulk_russell_binary_similarity,
        bulk_russell_binary_distance,
    )
