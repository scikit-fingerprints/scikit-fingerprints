from collections.abc import Sequence
from itertools import chain
from typing import Any, Optional, Union

import numpy as np
from sklearn.model_selection._split import _validate_shuffle_split
from sklearn.utils import _safe_indexing


def ensure_nonempty_subset(data: list, subset: str) -> None:
    """
    Check if the provided list is empty.
    """
    if len(data) == 0:
        raise ValueError(f"{subset.capitalize()} subset is empty")


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


def validate_train_test_split_sizes(
    train_size: Optional[Union[float, int]],
    test_size: Optional[Union[float, int]],
    n_samples: int,
) -> tuple[int, int]:
    """
    Ensure the sum of train_size and test_size equals 1.0 and provide default values
    if necessary. Returns integers with number of rows in those subsets.
    """
    return _validate_shuffle_split(
        n_samples, test_size, train_size, default_test_size=0.2
    )


def validate_train_valid_test_split_sizes(
    train_size: Optional[Union[float, int]],
    valid_size: Optional[Union[float, int]],
    test_size: Optional[Union[float, int]],
    n_samples: int,
) -> tuple[int, int, int]:
    """
    Ensure the sum of train_size, valid_size and test_size equals 1.0 and provide
    default values if necessary. Returns integers with number of rows in those subsets.
    """
    if train_size is None and valid_size is None and test_size is None:
        train_size = 0.8
        valid_size = 0.1
        test_size = 0.1

    sizes = (train_size, valid_size, test_size)
    if None in sizes:
        raise ValueError(f"All of the sizes must be provided, got: {sizes}")

    test_size_type = np.asarray(test_size).dtype.kind
    valid_size_type = np.asarray(valid_size).dtype.kind
    train_size_type = np.asarray(train_size).dtype.kind

    train_size: Union[int, float]
    valid_size: Union[int, float]
    test_size: Union[int, float]

    _check_subset_size(train_size, n_samples, "train")
    _check_subset_size(valid_size, n_samples, "valid")
    _check_subset_size(test_size, n_samples, "test")

    if (
        train_size_type == "f"
        and valid_size_type == "f"
        and test_size_type == "f"
        and train_size + valid_size + test_size > 1
    ):
        raise ValueError(
            f"The sum of float train_size, valid_size and test_size = "
            f"{train_size + valid_size + test_size}, should be in the (0, 1) "
            f"range."
        )

    n_test = int(test_size * n_samples) if test_size_type == "f" else test_size
    n_valid = int(valid_size * n_samples) if valid_size_type == "f" else valid_size
    n_train = n_samples - (n_test + n_valid) if valid_size_type == "f" else train_size

    if not n_train or not n_valid or not n_test:
        raise ValueError(
            f"With current sizes of train_size={train_size}, valid_size={valid_size}, "
            f"test_size={test_size}, and n_samples={n_samples}, one of the sets will "
            f"be empty."
        )
    if n_train + n_valid + n_test != n_samples:
        raise ValueError(
            f"The sum of train, valid and test sizes must be equal to the "
            f"n_samples={n_samples}, got: "
            f"n_train={n_train} (train_size={train_size}), "
            f"n_valid={n_valid} (valid_size={valid_size}), "
            f"n_test={n_test} (test_size={test_size})."
        )

    return int(n_train), int(n_valid), int(n_test)


def _check_subset_size(size: Union[float, int], n_samples: int, subset: str) -> None:
    size_type = np.asarray(size).dtype.kind
    subset_size_str = f"{subset}_size"  # e.g. "train_size", "test_size"

    if size_type not in ("i", "f"):
        raise ValueError(f"Invalid value for {subset_size_str}: {size}")

    if (size_type == "i" and (size >= n_samples or size <= 0)) or (
        size_type == "f" and (size <= 0 or size >= 1)
    ):
        raise ValueError(
            f"{subset_size_str}={size} should be either positive and smaller than "
            f"the number of samples {n_samples} or a float in the (0, 1) range"
        )
