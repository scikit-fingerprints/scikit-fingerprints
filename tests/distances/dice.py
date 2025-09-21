import pytest

from skfp.distances import (
    dice_binary_distance,
    dice_binary_similarity,
    dice_count_distance,
    dice_count_similarity,
)
from skfp.distances.dice import (
    bulk_dice_binary_distance,
    bulk_dice_binary_similarity,
    bulk_dice_count_distance,
    bulk_dice_count_similarity,
)
from tests.distances.utils import (
    run_test_bulk_similarity_and_distance,
    run_test_bulk_similarity_and_distance_two_arrays,
    run_test_similarity_and_distance,
)


def _get_binary_values() -> list[tuple[list[int], list[int], str, float, float]]:
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


def _get_count_values() -> list[tuple[list[int], list[int], str, float, float]]:
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


@pytest.mark.parametrize(
    "vec_a, vec_b, comparison, similarity, distance", _get_binary_values()
)
def test_dice_binary(vec_a, vec_b, comparison, similarity, distance):
    run_test_similarity_and_distance(
        dice_binary_similarity,
        dice_binary_distance,
        vec_a,
        vec_b,
        comparison,
        similarity,
        distance,
    )


@pytest.mark.parametrize(
    "vec_a, vec_b, comparison, similarity, distance", _get_count_values()
)
def test_dice_count(vec_a, vec_b, comparison, similarity, distance):
    run_test_similarity_and_distance(
        dice_count_similarity,
        dice_count_distance,
        vec_a,
        vec_b,
        comparison,
        similarity,
        distance,
    )


def test_bulk_dice_binary(mols_list):
    run_test_bulk_similarity_and_distance(
        mols_list,
        dice_binary_similarity,
        dice_binary_distance,
        bulk_dice_binary_similarity,
        bulk_dice_binary_distance,
    )


def test_bulk_dice_count(mols_list):
    run_test_bulk_similarity_and_distance(
        mols_list,
        dice_count_similarity,
        dice_count_distance,
        bulk_dice_count_similarity,
        bulk_dice_count_distance,
        count=True,
    )


def test_bulk_dice_second_array_binary(mols_list):
    run_test_bulk_similarity_and_distance_two_arrays(
        mols_list,
        dice_binary_similarity,
        dice_binary_distance,
        bulk_dice_binary_similarity,
        bulk_dice_binary_distance,
    )


def test_bulk_dice_second_array_count(mols_list):
    run_test_bulk_similarity_and_distance_two_arrays(
        mols_list,
        dice_count_similarity,
        dice_count_distance,
        bulk_dice_count_similarity,
        bulk_dice_count_distance,
        count=True,
    )
