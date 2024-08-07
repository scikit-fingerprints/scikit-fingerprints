import warnings
from collections.abc import Sequence
from itertools import chain
from typing import Any, Optional, Union

import numpy as np
from rdkit.Chem import Mol
from sklearn.utils import _safe_indexing


def ensure_nonempty_list(subset: list) -> None:
    """
    Check if the provided subset is empty.
    """
    if len(subset) == 0:
        raise ValueError("One of the subsets is empty.")


def validate_train_test_sizes(
    train_size: Optional[float], test_size: Optional[float]
) -> tuple[float, float]:
    """
    Fill in missing sizes for train and test sets based on the provided sizes.
    """
    if train_size is None and test_size is None:
        return 0.8, 0.2
    if train_size is None:
        if test_size is not None:
            train_size = 1 - test_size
        else:
            raise ValueError("test_size must be provided when train_size is None")
    elif test_size is None:
        test_size = 1 - train_size
    return train_size, test_size


def get_data_from_indices(
    data: Sequence[Union[str, Mol]], indices: Sequence[int]
) -> list[Union[str, Mol]]:
    """
    Helper function to retrieve data elements from specified indices.
    """
    return [data[idx] for idx in set(indices)]


def split_additional_data(
    additional_data: list[Sequence[Any]], *indices_lists: list[list[int]]
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
    train_size: Optional[float],
    valid_size: Optional[float],
    test_size: Optional[float],
) -> tuple[float, float, float]:
    """
    Ensure the sum of train_size, valid_size, and test_size equals 1.0 and provide default values if necessary.
    """
    if train_size is None and valid_size is None and test_size is None:
        return 0.8, 0.1, 0.1

    if train_size is None or valid_size is None or test_size is None:
        raise ValueError(
            "All of train_size, valid_size, and test_size must be provided."
        )

    if valid_size == 0.0:
        warnings.warn(
            "Validation set will not be returned since valid_size was set to 0.0."
            "Consider using train_test_split instead."
        )

    if not np.isclose(train_size + valid_size + test_size, 1.0):
        raise ValueError("train_size, test_size, and valid_size must sum to 1.0")

    return train_size, valid_size, test_size
