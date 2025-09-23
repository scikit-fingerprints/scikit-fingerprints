import pytest

from skfp.distances import (
    harris_lahey_binary_distance,
    harris_lahey_binary_similarity,
)
from skfp.distances.harris_lahey import (
    bulk_harris_lahey_binary_distance,
    bulk_harris_lahey_binary_similarity,
)
from tests.distances.utils import (
    run_test_bulk_similarity_and_distance,
    run_test_bulk_similarity_and_distance_two_arrays,
    run_test_similarity_and_distance,
)


def _get_unnormalized_values() -> list[tuple[list[int], list[int], float, float]]:
    # vec_a, vec_b, similarity, distance
    return [
        ([1, 0, 0], [0, 1, 1], 0.0, 1.0),
        ([1, 0, 0], [0, 0, 0], 1 / 3, 8 / 9),
        ([0, 0, 0], [0, 0, 0], 1.0, 0.0),
        ([1, 0, 0], [1, 0, 0], 3.0, 0.0),
        ([1, 1, 1], [1, 1, 1], 1.0, 0.0),
        ([1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1], 0.75, 0.875),
    ]


def _get_normalized_values() -> list[tuple[list[int], list[int], float, float]]:
    # vec_a, vec_b, similarity, distance
    return [
        ([1, 0, 0], [0, 1, 1], 0.0, 1.0),
        ([1, 0, 0], [0, 0, 0], 1 / 9, 8 / 9),
        ([0, 0, 0], [0, 0, 0], 1.0, 0.0),
        ([1, 0, 0], [1, 0, 0], 1.0, 0.0),
        ([1, 1, 1], [1, 1, 1], 1.0, 0.0),
        ([1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1], 0.125, 0.875),
    ]


@pytest.mark.parametrize(
    "vec_a, vec_b, similarity, distance", _get_unnormalized_values()
)
def test_harris_lahey_unnormalized(vec_a, vec_b, similarity, distance):
    run_test_similarity_and_distance(
        harris_lahey_binary_similarity,
        harris_lahey_binary_distance,
        vec_a,
        vec_b,
        similarity,
        distance,
        normalized=False,
    )


@pytest.mark.parametrize("vec_a, vec_b, similarity, distance", _get_normalized_values())
def test_harris_lahey_normalized(vec_a, vec_b, similarity, distance):
    run_test_similarity_and_distance(
        harris_lahey_binary_similarity,
        harris_lahey_binary_distance,
        vec_a,
        vec_b,
        similarity,
        distance,
        normalized=True,
    )


@pytest.mark.parametrize("normalized", [False, True])
def test_bulk_harris_lahey(mols_list, normalized):
    run_test_bulk_similarity_and_distance(
        mols_list,
        harris_lahey_binary_similarity,
        harris_lahey_binary_distance,
        bulk_harris_lahey_binary_similarity,
        bulk_harris_lahey_binary_distance,
        normalized=normalized,
    )


@pytest.mark.parametrize("normalized", [False, True])
def test_bulk_harris_lahey_second_array(mols_list, normalized):
    run_test_bulk_similarity_and_distance_two_arrays(
        mols_list,
        harris_lahey_binary_similarity,
        harris_lahey_binary_distance,
        bulk_harris_lahey_binary_similarity,
        bulk_harris_lahey_binary_distance,
        normalized=normalized,
    )
