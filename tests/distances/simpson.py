import pytest

from skfp.distances import simpson_binary_distance, simpson_binary_similarity
from skfp.distances.simpson import (
    bulk_simpson_binary_distance,
    bulk_simpson_binary_similarity,
)
from tests.distances.utils import (
    run_test_bulk_similarity_and_distance,
    run_test_bulk_similarity_and_distance_two_arrays,
    run_test_similarity_and_distance,
)


def _get_values() -> list[tuple[list[int], list[int], str, float, float]]:
    # vec_a, vec_b, comparison, similarity, distance
    return [
        ([1, 0, 0], [0, 1, 1], "==", 0.0, 1.0),
        ([1, 0, 0], [0, 0, 0], "==", 0.0, 1.0),
        ([0, 0, 0], [0, 0, 0], "==", 0.0, 1.0),
        ([1, 0, 0], [1, 0, 0], "==", 1.0, 0.0),
        ([1, 1, 1], [1, 1, 1], "==", 1.0, 0.0),
        ([1, 0, 0, 0], [1, 1, 1, 1], "==", 1.0, 0.0),
        ([1, 1, 1, 0], [1, 1, 1, 1], ">", 0.5, 0.5),
    ]


@pytest.mark.parametrize(
    "vec_a, vec_b, comparison, similarity, distance", _get_values()
)
def test_simpson(vec_a, vec_b, comparison, similarity, distance):
    run_test_similarity_and_distance(
        simpson_binary_similarity,
        simpson_binary_distance,
        vec_a,
        vec_b,
        comparison,
        similarity,
        distance,
    )


def test_bulk_simpson(mols_list):
    run_test_bulk_similarity_and_distance(
        mols_list,
        simpson_binary_similarity,
        simpson_binary_distance,
        bulk_simpson_binary_similarity,
        bulk_simpson_binary_distance,
    )


def test_bulk_simpson_second_array(mols_list):
    run_test_bulk_similarity_and_distance_two_arrays(
        mols_list,
        simpson_binary_similarity,
        simpson_binary_distance,
        bulk_simpson_binary_similarity,
        bulk_simpson_binary_distance,
    )
