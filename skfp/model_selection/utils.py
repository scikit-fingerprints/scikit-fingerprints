from collections.abc import Sequence
from itertools import chain
from typing import Any, Optional, Union

import numpy as np
from rdkit.Chem import Mol
from sklearn.utils import _safe_indexing


def ensure_nonempty_subset(data: list, subset: str) -> None:
    """
    Check if the provided list is empty.
    """
    if len(data) == 0:
        raise ValueError(f"{subset.capitalize()} subset is empty")


def validate_train_test_split_sizes(
    train_size: Optional[Union[float, int]],
    test_size: Optional[Union[float, int]],
    data_length: int,
) -> tuple[float, float]:
    """
    Fill in missing sizes for train and test sets based on the provided sizes.
    If test_size and train_size are floats, this method returns concrete
    number of rows, i.e. int(size * data_length)

    If test_size and train_size are ints, this method returns the provided sizes.
    """
    if train_size is None and test_size is None:
        train_size, test_size = 0.8, 0.2

    if (
        train_size is not None
        and test_size is not None
        and type(train_size) is not type(test_size)
    ):
        raise TypeError(
            f"train_size and test_size must be of the same type, got {type(train_size)} for "
            f"train_size and {type(test_size)} for test_size"
        )

    if train_size is None:
        train_size = 1 - test_size if test_size is not None else 0

    if isinstance(train_size, int) and train_size >= data_length:
        raise ValueError(
            f"train_size as an integer must be smaller than data_length, got {train_size} for "
            f"data_length {data_length}"
        )

    if test_size is None:
        test_size = 1 - train_size if train_size is not None else 0

    if isinstance(test_size, int) and test_size >= data_length:
        raise ValueError(
            f"test_size as an integer must be smaller than data_length, got {test_size} for "
            f"data_length {data_length}"
        )

    if isinstance(train_size, float) and not np.isclose(train_size + test_size, 1.0):
        raise ValueError("train_size and test_size must sum to 1.0")

    if train_size == 0.0:
        raise ValueError("train_size is 0.0")
    if test_size == 0.0:
        raise ValueError("test_size is 0.0")

    if isinstance(train_size, float):
        return (
            train_size * data_length,
            test_size * data_length,
        )
    else:
        return train_size, test_size


def get_data_from_indices(
    data: Sequence[Union[str, Mol]], indices: Sequence[int]
) -> list[Union[str, Mol]]:
    """
    Helper function to retrieve data elements from specified indices.
    """
    return [data[idx] for idx in set(indices)]


def split_additional_data(
    additional_data: list[Sequence[Any]], *indices_lists: list[int]
) -> list[Sequence[Any]]:
    """
    Split additional data based on indices lists.
    """
    return list(
        chain.from_iterable(
            (_safe_indexing(a, indices),)
            for a in additional_data
            for indices in indices_lists
        )
    )


def validate_train_valid_test_split_sizes(
    train_size: Optional[Union[float, int]],
    valid_size: Optional[Union[float, int]],
    test_size: Optional[Union[float, int]],
    data_length: int,
) -> tuple[float, float, float]:
    """
    Ensure the sum of train_size, valid_size, and test_size equals 1.0 and provide default values if necessary.
    If test_size, valid_size and train_size are floats, this method returns concrete
    number of rows, i.e. int(size * data_length)
    If test_size, valid_size and train_size are ints, this method returns the provided sizes.
    """
    if train_size is None and valid_size is None and test_size is None:
        train_size, valid_size, test_size = 0.8, 0.1, 0.1

    if train_size is None or valid_size is None or test_size is None:
        raise ValueError("All of the sizes must be provided")

    sizes: tuple[Union[float, int], Union[float, int], Union[float, int]] = (
        train_size,
        valid_size,
        test_size,
    )

    if not all(isinstance(size, (int, float)) for size in sizes):
        types = [type(size) for size in sizes]
        raise TypeError(f"All sizes must be either int or float, got: {types}")

    if not all(isinstance(size, type(train_size)) for size in sizes):
        types = [type(size) for size in sizes]
        raise TypeError(f"All sizes must be of the same type, got: {types}")

    if isinstance(train_size, float):
        if not np.isclose(sum(sizes), 1.0):
            raise ValueError(
                f"The sum of train_size, valid_size, and test_size must be 1.0, got: "
                f"{sum(sizes)}"
            )

        return (
            train_size * data_length,
            valid_size * data_length,
            test_size * data_length,
        )

    if isinstance(train_size, int):
        total = sum(sizes)
        if total != data_length:
            raise ValueError(
                f"The sum of train_size, valid_size, and test_size must equal data_length, "
                f"got {total} instead"
            )

        return train_size, valid_size, test_size
