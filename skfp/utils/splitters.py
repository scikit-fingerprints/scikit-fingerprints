import numbers
import warnings
from collections import defaultdict
from collections.abc import Sequence
from itertools import chain
from numbers import Integral
from typing import Any, Optional, Union

import numpy as np
from rdkit.Chem import Mol
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.utils import _safe_indexing
from sklearn.utils._param_validation import Interval, RealNotInt, validate_params

from skfp.utils.validators import ensure_mols


@validate_params(
    {
        "data": ["sequence"],
        "additional_data": ["list"],
        "train_size": [
            Interval(RealNotInt, 0, 1, closed="neither"),
            Interval(Integral, 1, float("inf"), closed="left"),
            None,
        ],
        "test_size": [
            Interval(RealNotInt, 0, 1, closed="neither"),
            Interval(Integral, 1, float("inf"), closed="left"),
            None,
        ],
        "include_chirality": ["boolean"],
        "return_indices": ["boolean"],
    }
)
def scaffold_train_test_split(
    data: Sequence[Union[str, Mol]],
    *additional_data: Sequence,
    train_size: Optional[float] = None,
    test_size: Optional[float] = None,
    include_chirality: bool = False,
    return_indices: bool = False,
) -> Union[
    tuple[list[Union[str, Mol]], list[Union[str, Mol]], list[Sequence[Any]]],
    tuple[list[int], list[int], list[Sequence[Any]]],
    tuple[list[int], list[int]],
]:
    """
    Split a list of SMILES or RDKit `Mol` objects into train and test subsets using Bemis-Murcko [1]_ scaffolds.

    This function ensures that similar molecules (sharing the same scaffold) are in the same split.
    This method, suggested in the MoleculeNet [2]_ paper, helps to test the model's ability to
    generalize to entirely new scaffolds. MoleculeNet introduced the scaffold split as an approximation
    to the time split, assuming that new molecules (test set) will be structurally different in terms of
    scaffolds from the training set.
    Note that there are limitations to this functionality. For example, disconnected molecules or
    molecules with no rings will not get a scaffold, resulting in them being grouped together
    regardless of their structure.

    The split is fully deterministic, with the smallest scaffold sets assigned to the test
    subset and the rest to the training subset.

    The split fractions (train_size, test_size) must sum to 1.

    Parameters
    ----------
    data : sequence
        Sequence representing either SMILES strings or RDKit `Mol` objects.

    additional_data: list[sequence]
        Additional sequences to be split alongside the main data (e.g., labels or feature vectors).

    train_size : float, default=None
        The fraction of data to be used for the train subset. If None, it is set to 1 - test_size.

    test_size : float, default=None
        The fraction of data to be used for the test subset. If None, it is set to 1 - train_size.

    include_chirality: bool, default=False
        Whether to take chirality of molecules into consideration.

    return_indices : bool, default=False
        Whether the method should return the input object subsets, i.e. SMILES strings
        or RDKit `Mol` objects, or only the indices of the subsets.

    Returns
    ----------
    subsets : tuple[list, list]
        A tuple of train and test subsets. The format depends on `return_indices`:
        - if `return_indices` is False, returns lists of SMILES strings
        or RDKit `Mol` objects depending on the input
        - if `return_indices` is True, returns lists of indices
    additional_data_splitted: list[sequence]:
        the method may return any additional data splitted in the same way as
        provided SMILES

    References
    ----------
    .. [1] `Bemis, G. W., & Murcko, M. A.
        "The properties of known drugs. 1. Molecular frameworks."
        Journal of Medicinal Chemistry, 39(15), 2887-2893.
        https://www.researchgate.net/publication/14493474_The_Properties_of_Known_Drugs_1_Molecular_Frameworks`_

    .. [2] `Z. Wu, B. Ramsundar, E. N. Feinberg, J. Gomes, C. Geniesse, A. S. Pappu, K. Leswing, V. Pande
        "MoleculeNet: A Benchmark for Molecular Machine Learning."
        Chemical Science, 9(2), 513-530.
        https://www.researchgate.net/publication/314182452_MoleculeNet_A_Benchmark_for_Molecular_Machine_Learning`_

    """
    _validate_split_sizes(train_size, test_size)
    train_size, test_size = _fill_missing_sizes(train_size, test_size)
    scaffolds = _create_scaffolds(list(data), include_chirality)
    scaffold_sets = sorted(scaffolds.values(), key=len)

    train_ids: list[int] = []
    test_ids: list[int] = []

    train_ids, test_ids = _split_ids(scaffold_sets, len(data), test_size)

    train_subset: list[Any] = []
    test_subset: list[Any] = []

    if return_indices:
        train_subset = train_ids
        test_subset = test_ids
    else:
        train_subset = _get_data_from_indices(data, train_ids)
        test_subset = _get_data_from_indices(data, test_ids)

    _check_subsets(train_subset, test_subset)

    additional_data_split: list[Sequence[Any]] = []

    additional_data_split = (
        _split_additional_data(list(additional_data), [train_ids, test_ids])
        if additional_data
        else []
    )

    if additional_data:
        return train_subset, test_subset, additional_data_split
    else:
        return train_subset, test_subset


@validate_params(
    {
        "data": ["sequence"],
        "additional_data": ["list"],
        "train_size": [
            Interval(RealNotInt, 0, 1, closed="neither"),
            Interval(Integral, 1, float("inf"), closed="left"),
            None,
        ],
        "valid_size": [
            Interval(RealNotInt, 0, 1, closed="neither"),
            Interval(Integral, 1, float("inf"), closed="left"),
            None,
        ],
        "test_size": [
            Interval(RealNotInt, 0, 1, closed="neither"),
            Interval(Integral, 1, float("inf"), closed="left"),
            None,
        ],
        "include_chirality": ["boolean"],
        "return_indices": ["boolean"],
    }
)
def scaffold_train_valid_test_split(
    data: Sequence[Union[str, Mol]],
    *additional_data: Sequence,
    train_size: Optional[float] = None,
    test_size: Optional[float] = None,
    valid_size: Optional[float] = None,
    include_chirality: bool = False,
    return_indices: bool = False,
) -> Union[
    tuple[list[Union[str, Mol]], list[Union[str, Mol]], list[Union[str, Mol]]],
    tuple[list[int], list[int], list[int]],
    tuple[
        list[Union[str, Mol]],
        list[Union[str, Mol]],
        list[Union[str, Mol]],
        list[Sequence[Any]],
    ],
    tuple[list[int], list[int], list[int], list[Sequence[Any]]],
]:
    """
    Split a list of SMILES or RDKit `Mol` objects into train and test subsets using Bemis-Murcko [1]_ scaffolds.

    This function ensures that similar molecules (sharing the same scaffold) are in the same split.
    This method, suggested in the MoleculeNet [2]_ paper, helps to test the model's ability to
    generalize to entirely new scaffolds. MoleculeNet introduced the scaffold split as an approximation
    to the time split, assuming that new molecules (test set) will be structurally different in terms of
    scaffolds from the training set.
    Note that there are limitations to this functionality. For example, disconnected molecules or
    molecules with no rings will not get a scaffold, resulting in them being grouped together
    regardless of their structure.

    The split is fully deterministic, with the smallest scaffold sets assigned to the test or validation
    subset and the rest to the training subset.

    The split fractions (train_size, test_size) must sum to 1.

    Parameters
    ----------
    data : sequence
        Sequence representing either SMILES strings or RDKit `Mol` objects.

    additional_data: sequence
        Additional sequences to be split alongside the main data (e.g., labels or feature vectors).

    train_size : float, default=None
        The fraction of data to be used for the train subset. If None, it is set to 1 - test_size - valid_size.
        If valid_size is not provided, train_size is set to 1 - test_size. If train_size, test_size and
        valid_size aren't set, train_size is set to 0.8.

    test_size : float, default=None
        The fraction of data to be used for the validation subset. If None, it is set to 1 - train_size - valid_size.
        If valid_size is not provided, test_size is set to 1 - train_size. If train_size, test_size and
        valid_size aren't set, test_size is set to 0.1.

    valid_size : float, default=None
        The fraction of data to be used for the test subset. If None, it is set to 1 - train_size - valid_size.
        If train_size, test_size and valid_size aren't set, train_size is set to 0.1.

    include_chirality: bool, default=False
        Whether to take chirality of molecules into consideration.

    return_indices : bool, default=False
        Whether the method should return the input object subsets, i.e. SMILES strings
        or RDKit `Mol` objects, or only the indices of the subsets.

    Returns
    ----------
    subsets : tuple[list, list, list]
        A tuple of train, validation, and test subsets. The format depends on `return_indices`:
        - if `return_indices` is False, returns lists of SMILES strings
        or RDKit `Mol` objects depending on the input
        - if `return_indices` is True, returns lists of indices
    additional_data_splitted: list[sequence]:
        the method may return any additional data splitted in the same way as
        provided SMILES

    References
    ----------
    .. [1] `Bemis, G. W., & Murcko, M. A.
        "The properties of known drugs. 1. Molecular frameworks."
        Journal of Medicinal Chemistry, 39(15), 2887-2893.
        https://www.researchgate.net/publication/14493474_The_Properties_of_Known_Drugs_1_Molecular_Frameworks`_

    .. [2] `Z. Wu, B. Ramsundar, E. N. Feinberg, J. Gomes, C. Geniesse, A. S. Pappu, K. Leswing, V. Pande
        "MoleculeNet: A Benchmark for Molecular Machine Learning."
        Chemical Science, 9(2), 513-530.
        https://www.researchgate.net/publication/314182452_MoleculeNet_A_Benchmark_for_Molecular_Machine_Learning`_

    """
    train_size, valid_size, test_size = _split_size(train_size, valid_size, test_size)

    scaffolds = _create_scaffolds(list(data), include_chirality)
    scaffold_sets = sorted(scaffolds.values(), key=len)

    train_ids: list[int] = []
    valid_ids: list[int] = []
    test_ids: list[int] = []

    train_ids, valid_ids, test_ids = _split_ids_three_sets(
        scaffold_sets, len(data), test_size, valid_size
    )

    train_subset: list[Any] = []
    valid_subset: list[Any] = []
    test_subset: list[Any] = []

    if return_indices:
        train_subset = train_ids
        valid_subset = valid_ids
        test_subset = test_ids
    else:
        train_subset = _get_data_from_indices(data, train_ids)
        valid_subset = _get_data_from_indices(data, valid_ids)
        test_subset = _get_data_from_indices(data, test_ids)

    _check_subsets(train_subset, test_subset)

    if len(valid_subset) == 0:
        warnings.warn(
            "Warning: Valid subset is empty. Consider using scaffold_train_test_split instead."
        )

    additional_data_split: list[Sequence[Any]] = []

    additional_data_split = (
        _split_additional_data(list(additional_data), [train_ids, valid_ids, test_ids])
        if additional_data
        else []
    )

    if additional_data:
        return train_subset, valid_subset, test_subset, additional_data_split
    else:
        return train_subset, valid_subset, test_subset


def _calculate_missing_sizes(
    train_size: Optional[float],
    valid_size: Optional[float],
    test_size: Optional[float],
) -> tuple[float, float, float]:
    """
    Calculate the missing sizes for train, validation, and test sets if they are not provided.
    """
    if train_size is None and test_size is None and valid_size is None:
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

    return train_size, test_size, valid_size


def _check_subsets(*subsets: list) -> None:
    """
    Check if any of the provided subsets is empty.
    """
    for subset in subsets:
        if len(subset) == 0:
            raise ValueError("One of the subsets is empty.")


def _create_scaffolds(
    data: list[Union[str, Mol]], include_chirality: bool = False
) -> dict[str, list]:
    """
    Generate Bemis-Murcko scaffolds for a list of SMILES strings or RDKit `Mol` objects.
    This implementation uses Bemis-Murcko scaffolds [1]_ to group molecules.
    Each scaffold is represented as a SMILES string.
    """
    scaffolds = defaultdict(list)
    molecules = ensure_mols(data)

    for ind, molecule in enumerate(molecules):
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=molecule, includeChirality=include_chirality
        )
        scaffolds[scaffold].append(ind)

    return scaffolds


def _fill_missing_sizes(
    train_size: Optional[float], test_size: Optional[float]
) -> tuple[float, float]:
    """
    Fill in missing sizes for train and test sets based on the provided sizes.
    """
    if train_size is None and test_size is None:
        train_size = 0.8
        test_size = 0.2
    if train_size is None:
        if test_size is not None:
            train_size = 1 - test_size
        else:
            raise ValueError("test_size must be provided when train_size is None")
    elif test_size is None:
        test_size = 1 - train_size
    return train_size, test_size


def _get_data_from_indices(
    data: Sequence[Union[str, Mol]], indices: Sequence[int]
) -> list[Union[str, Mol]]:
    """
    Helper function to retrieve data elements from specified indices.
    """
    indices = set(indices)
    return [data[idx] for idx in indices]


def _split_additional_data(
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


def _split_ids(
    scaffold_sets: list[list[int]], total_data_len: int, test_size: float
) -> tuple[list[int], list[int]]:
    """
    Split IDs into training and testing sets based on scaffold sets.
    """
    train_ids: list[int] = []
    test_ids: list[int] = []
    desired_test_size = int(test_size * total_data_len)

    for scaffold_set in scaffold_sets:
        if len(test_ids) < desired_test_size:
            test_ids.extend(scaffold_set)
        else:
            train_ids.extend(scaffold_set)

    return train_ids, test_ids


def _split_ids_three_sets(
    scaffold_sets: list[list[int]],
    total_data_len: int,
    test_size: float,
    valid_size: float,
) -> tuple[list[int], list[int], list[int]]:
    """
    Split IDs into training, validation, and testing sets based on scaffold sets.
    """
    train_ids: list[int] = []
    valid_ids: list[int] = []
    test_ids: list[int] = []
    desired_test_size = int(test_size * total_data_len)
    desired_valid_size = int((test_size + valid_size) * total_data_len)

    for scaffold_set in scaffold_sets:
        if len(test_ids) < desired_test_size:
            test_ids.extend(scaffold_set)
        elif len(valid_ids) < desired_valid_size:
            valid_ids.extend(scaffold_set)
        else:
            train_ids.extend(scaffold_set)

    return train_ids, valid_ids, test_ids


def _split_size(
    train_size: Optional[float],
    valid_size: Optional[float],
    test_size: Optional[float],
) -> tuple[float, float, float]:
    """
    Ensure the sum of train_size, valid_size, and test_size equals 1.0 and provide default values if necessary.
    """
    train_size, test_size, valid_size = _calculate_missing_sizes(
        train_size, test_size, valid_size
    )

    if not np.isclose(train_size + test_size + valid_size, 1.0):
        raise ValueError("train_size, test_size, and valid_size must sum to 1.0")

    return train_size, test_size, valid_size


def _validate_split_sizes(
    train_size: Optional[float], test_size: Optional[float]
) -> None:
    """
    Validate that the provided train_size and test_size are correct and sum to 1.0.
    """
    if train_size is None and test_size is None:
        raise ValueError("Either train_size or test_size must be provided")
    if (
        train_size is not None
        and test_size is not None
        and not np.isclose(train_size + test_size, 1.0)
    ):
        raise ValueError("train_size and test_size must sum to 1.0")
