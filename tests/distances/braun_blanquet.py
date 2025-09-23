import pytest

from skfp.distances import (
    braun_blanquet_binary_distance,
    braun_blanquet_binary_similarity,
)
from skfp.distances.braun_blanquet import (
    bulk_braun_blanquet_binary_distance,
    bulk_braun_blanquet_binary_similarity,
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
        ([0, 0, 0], [0, 0, 0], 1.0, 0.0),
        ([1, 0, 0], [1, 0, 0], 1.0, 0.0),
        ([1, 1, 1], [1, 1, 1], 1.0, 0.0),
        ([1, 0, 0, 0], [1, 1, 1, 1], 0.25, 0.75),
        ([1, 1, 1, 0], [1, 1, 1, 1], 0.75, 0.25),
    ]


@pytest.mark.parametrize("vec_a, vec_b, similarity, distance", _get_values())
def test_braun_blanquet(vec_a, vec_b, similarity, distance):
    run_test_similarity_and_distance(
        braun_blanquet_binary_similarity,
        braun_blanquet_binary_distance,
        vec_a,
        vec_b,
        similarity,
        distance,
    )


def test_bulk_braun_blanquet(mols_list):
    run_test_bulk_similarity_and_distance(
        mols_list,
        braun_blanquet_binary_similarity,
        braun_blanquet_binary_distance,
        bulk_braun_blanquet_binary_similarity,
        bulk_braun_blanquet_binary_distance,
    )


def test_bulk_braun_blanquet_second_array(mols_list):
    run_test_bulk_similarity_and_distance_two_arrays(
        mols_list,
        braun_blanquet_binary_similarity,
        braun_blanquet_binary_distance,
        bulk_braun_blanquet_binary_similarity,
        bulk_braun_blanquet_binary_distance,
    )
