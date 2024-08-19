from typing import Union

import pytest
from rdkit.Chem import Mol

from skfp.model_selection.utils import (
    ensure_nonempty_list,
    get_data_from_indices,
    split_additional_data,
    validate_train_test_sizes,
    validate_train_valid_test_split_sizes,
)


@pytest.fixture
def smiles_data() -> list[str]:
    return ["CCC", "CCCl", "CCO", "CCN"]


@pytest.fixture
def additional_data() -> list[list[Union[str, int, bool]]]:
    return [["a", "b", "c", "d"], [1, 2, 3, 4], [True, False, True, False]]


def test_ensure_nonempty_list_passes():
    ensure_nonempty_list([1, 2, 3])


def test_ensure_nonempty_list_raises_error():
    with pytest.raises(ValueError, match="Provided list is empty."):
        ensure_nonempty_list([])


def test_validate_train_test_sizes_both_provided():
    assert validate_train_test_sizes(0.7, 0.3) == (0.7, 0.3)


def test_validate_train_test_sizes_train_missing():
    assert validate_train_test_sizes(None, 0.3) == (0.7, 0.3)


def test_validate_train_test_sizes_test_missing():
    assert validate_train_test_sizes(0.6, None) == (0.6, 0.4)


def test_validate_train_test_sizes_both_missing():
    assert validate_train_test_sizes(None, None) == (0.8, 0.2)


def test_validate_train_test_sizes_not_sum_to_one():
    with pytest.raises(ValueError, match="train_size and test_size must sum to 1.0"):
        validate_train_test_sizes(0.6, 0.5)


def test_get_data_from_indices_valid(smiles_data):
    result = get_data_from_indices(smiles_data, [0, 2])
    assert result == ["CCC", "CCO"]


def test_get_data_from_indices_duplicates(smiles_data):
    # This works since we iterate over a set of indices
    result = get_data_from_indices(smiles_data, [0, 2, 2])
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
    result = validate_train_valid_test_split_sizes(0.7, 0.2, 0.1)
    assert result == (0.7, 0.2, 0.1)


def test_validate_train_valid_test_split_sizes_not_sum_to_one():
    with pytest.raises(
        ValueError, match="train_size, test_size, and valid_size must sum to 1.0"
    ):
        validate_train_valid_test_split_sizes(0.7, 0.2, 0.2)


def test_validate_train_valid_test_split_sizes_missing_values():
    with pytest.raises(
        ValueError,
        match="All of train_size, valid_size, and test_size must be provided.",
    ):
        validate_train_valid_test_split_sizes(0.7, None, 0.2)


def test_validate_train_valid_test_split_sizes_all_missing():
    result = validate_train_valid_test_split_sizes(None, None, None)
    assert result == (0.8, 0.1, 0.1)
