import pytest

from skfp.distances import mcconnaughey_binary_distance, mcconnaughey_binary_similarity
from skfp.distances.mcconnaughey import (
    bulk_mcconnaughey_binary_distance,
    bulk_mcconnaughey_binary_similarity,
)
from tests.distances.utils import (
    run_test_bulk_similarity_and_distance,
    run_test_bulk_similarity_and_distance_two_arrays,
    run_test_similarity_and_distance,
)


def _get_unnormalized_values() -> list[tuple[list[int], list[int], str, float, float]]:
    # vec_a, vec_b, comparison, similarity, distance
    return [
        ([1, 0, 0], [0, 1, 1], "==", -1.0, 1.0),
        ([0, 0, 0], [0, 0, 0], "==", 1.0, 0.0),
        ([1, 0, 0], [1, 0, 0], "==", 1.0, 0.0),
        ([1, 1, 1], [1, 1, 1], "==", 1.0, 0.0),
    ]


def _get_normalized_values() -> list[tuple[list[int], list[int], str, float, float]]:
    # vec_a, vec_b, comparison, similarity, distance
    return [
        ([1, 0, 0], [0, 1, 1], "==", 0.0, 1.0),
        ([1, 0, 0], [0, 0, 0], "==", 0.0, 1.0),
        ([0, 0, 0], [0, 0, 0], "==", 1.0, 0.0),
        ([1, 0, 0], [1, 0, 0], "==", 1.0, 0.0),
        ([1, 1, 1], [1, 1, 1], "==", 1.0, 0.0),
        ([1, 1, 1, 0], [1, 1, 1, 1], ">", 0.5, 0.5),
    ]


@pytest.mark.parametrize(
    "vec_a, vec_b, comparison, similarity, distance", _get_unnormalized_values()
)
def test_mcconnaughey_unnormalized(vec_a, vec_b, comparison, similarity, distance):
    run_test_similarity_and_distance(
        mcconnaughey_binary_similarity,
        mcconnaughey_binary_distance,
        vec_a,
        vec_b,
        comparison,
        similarity,
        distance,
        normalized=False,
    )


@pytest.mark.parametrize(
    "vec_a, vec_b, comparison, similarity, distance", _get_normalized_values()
)
def test_mcconnaughey_normalized(vec_a, vec_b, comparison, similarity, distance):
    run_test_similarity_and_distance(
        mcconnaughey_binary_similarity,
        mcconnaughey_binary_distance,
        vec_a,
        vec_b,
        comparison,
        similarity,
        distance,
        normalized=True,
    )


@pytest.mark.parametrize("normalized", [False, True])
def test_bulk_mcconnaughey(mols_list, normalized):
    run_test_bulk_similarity_and_distance(
        mols_list,
        mcconnaughey_binary_similarity,
        mcconnaughey_binary_distance,
        bulk_mcconnaughey_binary_similarity,
        bulk_mcconnaughey_binary_distance,
        normalized=normalized,
    )


@pytest.mark.parametrize("normalized", [False, True])
def test_bulk_mcconnaughey_second_array(mols_list, normalized):
    run_test_bulk_similarity_and_distance_two_arrays(
        mols_list,
        mcconnaughey_binary_similarity,
        mcconnaughey_binary_distance,
        bulk_mcconnaughey_binary_similarity,
        bulk_mcconnaughey_binary_distance,
        normalized=normalized,
    )
