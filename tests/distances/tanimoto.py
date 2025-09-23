import pytest

from skfp.distances import (
    bulk_tanimoto_binary_distance,
    bulk_tanimoto_binary_similarity,
    bulk_tanimoto_count_distance,
    bulk_tanimoto_count_similarity,
    tanimoto_binary_distance,
    tanimoto_binary_similarity,
    tanimoto_count_distance,
    tanimoto_count_similarity,
)
from tests.distances.utils import (
    run_test_bulk_similarity_and_distance,
    run_test_bulk_similarity_and_distance_two_arrays,
    run_test_similarity_and_distance,
)


def _get_binary_values() -> list[tuple[list[int], list[int], float, float]]:
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


def _get_count_values() -> list[tuple[list[int], list[int], float, float]]:
    # vec_a, vec_b, similarity, distance
    return [
        ([1, 0, 0], [0, 2, 3], 0.0, 1.0),
        ([1, 0, 0], [0, 0, 0], 0.0, 1.0),
        ([0, 0, 0], [0, 0, 0], 1.0, 0.0),
        ([1, 0, 0], [1, 0, 0], 1.0, 0.0),
        ([4, 0, 0], [4, 0, 0], 1.0, 0.0),
        ([1, 1, 1], [1, 1, 1], 1.0, 0.0),
        ([3, 2, 1], [3, 2, 1], 1.0, 0.0),
        ([3, 0, 0, 0], [1, 1, 1, 1], 0.3, 0.7),
        ([1, 3, 4, 0], [2, 3, 4, 2], 0.84375, 0.15625),
    ]


@pytest.mark.parametrize("vec_a, vec_b, similarity, distance", _get_binary_values())
def test_tanimoto_binary(vec_a, vec_b, similarity, distance):
    run_test_similarity_and_distance(
        tanimoto_binary_similarity,
        tanimoto_binary_distance,
        vec_a,
        vec_b,
        similarity,
        distance,
    )


@pytest.mark.parametrize("vec_a, vec_b, similarity, distance", _get_count_values())
def test_tanimoto_count(vec_a, vec_b, similarity, distance):
    run_test_similarity_and_distance(
        tanimoto_count_similarity,
        tanimoto_count_distance,
        vec_a,
        vec_b,
        similarity,
        distance,
    )


def test_bulk_tanimoto_binary(mols_list):
    run_test_bulk_similarity_and_distance(
        mols_list,
        tanimoto_binary_similarity,
        tanimoto_binary_distance,
        bulk_tanimoto_binary_similarity,
        bulk_tanimoto_binary_distance,
    )


def test_bulk_tanimoto_count(mols_list):
    run_test_bulk_similarity_and_distance(
        mols_list,
        tanimoto_count_similarity,
        tanimoto_count_distance,
        bulk_tanimoto_count_similarity,
        bulk_tanimoto_count_distance,
        count=True,
    )


def test_bulk_tanimoto_second_array_binary(mols_list):
    run_test_bulk_similarity_and_distance_two_arrays(
        mols_list,
        tanimoto_binary_similarity,
        tanimoto_binary_distance,
        bulk_tanimoto_binary_similarity,
        bulk_tanimoto_binary_distance,
    )


def test_bulk_tanimoto_second_array_count(mols_list):
    run_test_bulk_similarity_and_distance_two_arrays(
        mols_list,
        tanimoto_count_similarity,
        tanimoto_count_distance,
        bulk_tanimoto_count_similarity,
        bulk_tanimoto_count_distance,
        count=True,
    )
