import pytest

from skfp.distances import (
    rogot_goldberg_binary_distance,
    rogot_goldberg_binary_similarity,
)
from skfp.distances.rogot_goldberg import (
    bulk_rogot_goldberg_binary_distance,
    bulk_rogot_goldberg_binary_similarity,
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
        ([1, 0, 0], [0, 0, 0], 0.4, 0.6),
        ([0, 0, 0], [0, 0, 0], 1.0, 0.0),
        ([1, 0, 0], [1, 0, 0], 1.0, 0.0),
        ([1, 1, 1], [1, 1, 1], 1.0, 0.0),
        ([1, 1, 0, 0], [1, 1, 1, 1], 1 / 3, 2 / 3),
    ]


@pytest.mark.parametrize("vec_a, vec_b, similarity, distance", _get_values())
def test_rogot_goldberg(vec_a, vec_b, similarity, distance):
    run_test_similarity_and_distance(
        rogot_goldberg_binary_similarity,
        rogot_goldberg_binary_distance,
        vec_a,
        vec_b,
        similarity,
        distance,
    )


def test_bulk_rogot_goldberg(mols_list):
    run_test_bulk_similarity_and_distance(
        mols_list,
        rogot_goldberg_binary_similarity,
        rogot_goldberg_binary_distance,
        bulk_rogot_goldberg_binary_similarity,
        bulk_rogot_goldberg_binary_distance,
    )


def test_bulk_rogot_goldberg_second_array(mols_list):
    run_test_bulk_similarity_and_distance_two_arrays(
        mols_list,
        rogot_goldberg_binary_similarity,
        rogot_goldberg_binary_distance,
        bulk_rogot_goldberg_binary_similarity,
        bulk_rogot_goldberg_binary_distance,
    )
