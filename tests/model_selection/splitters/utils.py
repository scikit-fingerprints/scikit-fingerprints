import pytest

from skfp.model_selection.splitters.utils import (
    ensure_nonempty_subset,
    split_additional_data,
    validate_train_test_split_sizes,
    validate_train_valid_test_split_sizes,
)
from skfp.utils.functions import get_data_from_indices


@pytest.fixture
def smiles_data() -> list[str]:
    return ["CCC", "CCCl", "CCO", "CCN"]


@pytest.fixture
def additional_data() -> list[list[str | int | bool]]:
    return [["a", "b", "c", "d"], [1, 2, 3, 4], [True, False, True, False]]


def test_ensure_nonempty_subset_passes():
    ensure_nonempty_subset([1, 2, 3], "Test")


def test_ensure_nonempty_subset_raises_error():
    with pytest.raises(ValueError, match="Train subset is empty"):
        ensure_nonempty_subset([], "Train")


def test_validate_train_test_split_sizes_both_provided():
    assert validate_train_test_split_sizes(0.7, 0.3, 10) == (7, 3)


def test_validate_train_test_split_sizes_train_missing():
    assert validate_train_test_split_sizes(None, 0.3, 10) == (7, 3)


def test_validate_train_test_split_sizes_test_missing():
    assert validate_train_test_split_sizes(0.6, None, 10) == (6, 4)


def test_validate_train_test_split_sizes_both_missing():
    assert validate_train_test_split_sizes(None, None, 10) == (8, 2)


def test_validate_train_test_split_sizes_not_sum_to_one():
    with pytest.raises(ValueError, match="The sum of test_size and train_size"):
        validate_train_test_split_sizes(0.6, 0.5, 10)


def test_get_data_from_indices_valid(smiles_data):
    result = get_data_from_indices(smiles_data, [0, 2])
    assert result == ["CCC", "CCO"]


def test_get_data_from_indices_empty(smiles_data):
    result = get_data_from_indices(smiles_data, [])
    assert result == []


def test_get_data_from_indices_out_of_range(smiles_data):
    with pytest.raises(IndexError):
        get_data_from_indices(smiles_data, [0, 4])


def test_get_data_from_indices_mixed_types(smiles_data):
    result = get_data_from_indices(smiles_data, [0, 1, 2, 3])
    assert result == ["CCC", "CCCl", "CCO", "CCN"]


def test_split_additional_data(additional_data):
    result = split_additional_data(additional_data, [0, 2])
    assert result == [["a", "c"], [1, 3], [True, True]]


def test_split_additional_data_multiple_indices(additional_data):
    result = split_additional_data(additional_data, [0, 1], [2, 3])
    assert result == [
        ["a", "b"],
        ["c", "d"],
        [1, 2],
        [3, 4],
        [True, False],
        [True, False],
    ]


def test_split_additional_data_varying_lists_lengths(additional_data):
    result = split_additional_data(additional_data, [1], [0, 3])
    assert result == [["b"], ["a", "d"], [2], [1, 4], [False], [True, False]]


def test_split_additional_data_multiple_empty_indice_list(additional_data):
    result = split_additional_data(additional_data, [], [])
    assert result == [[], [], [], [], [], []]


def test_validate_train_valid_test_split_sizes_all_provided():
    result = validate_train_valid_test_split_sizes(0.7, 0.2, 0.1, 10)
    assert result == (7, 2, 1)


def test_validate_train_valid_test_split_sizes_not_sum_to_one():
    with pytest.raises(
        ValueError,
        match="The sum of float train_size, valid_size and test_size",
    ):
        validate_train_valid_test_split_sizes(0.7, 0.2, 0.2, 10)


def test_validate_train_valid_test_split_sizes_missing_values():
    with pytest.raises(
        ValueError,
        match="All of the sizes must be provided",
    ):
        validate_train_valid_test_split_sizes(0.7, None, 0.2, 10)


def test_validate_train_valid_test_split_sizes_all_missing():
    result = validate_train_valid_test_split_sizes(None, None, None, 10)
    assert result == (8, 1, 1)


def test_validate_train_test_split_sizes_different_type():
    result = validate_train_test_split_sizes(7, 0.3, 10)
    assert result == (7, 3)


def test_validate_train_valid_test_split_sizes_different_type():
    result = validate_train_valid_test_split_sizes(6, 0.3, 1, 10)
    assert result == (6, 3, 1)


def test_validate_train_test_split_sizes_train_size_too_large():
    with pytest.raises(
        ValueError,
        match="train_size=11 should be either positive and smaller than the number of samples",
    ):
        validate_train_test_split_sizes(11, 1, 10)


def test_validate_train_test_split_sizes_test_size_too_large():
    with pytest.raises(
        ValueError,
        match="test_size=11 should be either positive and smaller than the number of samples",
    ):
        validate_train_test_split_sizes(1, 11, 10)


def test_validate_train_test_split_sizes_train_size_zero():
    with pytest.raises(
        ValueError,
        match="test_size=1.0 should be either positive and smaller than the number of samples",
    ):
        validate_train_test_split_sizes(0.0, 1.0, 10)


def test_validate_train_test_split_sizes_test_size_zero():
    with pytest.raises(
        ValueError,
        match="test_size=0.0 should be either positive and smaller than the number of samples",
    ):
        validate_train_test_split_sizes(1.0, 0.0, 10)


def test_validate_train_valid_test_split_sizes_sum_not_equal_data_length_incorrect_total():
    with pytest.raises(
        ValueError,
        match="The sum of train, valid and test sizes must be equal to the n_samples=13",
    ):
        validate_train_valid_test_split_sizes(5, 3, 4, 13)
