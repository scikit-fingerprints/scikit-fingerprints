import warnings
from collections import defaultdict
from collections.abc import Sequence
from itertools import chain
from numbers import Integral
from typing import Any, Optional, Union

import numpy as np
from rdkit.Chem import Mol
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.utils._param_validation import Interval, RealNotInt, validate_params

from skfp.model_selection.utils import (
    ensure_nonempty_list,
    fill_missing_sizes,
    get_data_from_indices,
    split_additional_data,
    validate_train_valid_test_split_sizes,
)
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
    tuple[
        Sequence[Union[str, Mol]], Sequence[Union[str, Mol]], Sequence[Sequence[Any]]
    ],
    tuple[Sequence, ...],
    tuple[Sequence[int], Sequence[int]],
]:
    """
    Split a list of SMILES or RDKit `Mol` objects into train and test subsets using Bemis-Murcko [1]_ scaffolds.

    This function ensures that similar molecules (sharing the same scaffold) are in the same split.
    This method, suggested in the MoleculeNet [2]_ paper, helps to test the model's ability to
    generalize to entirely new scaffolds. MoleculeNet introduced the scaffold split as an approximation
    to the time split, assuming that new molecules (test set) will be structurally different in terms of
    scaffolds from the training set.

    This approach is known to have certain limitations. In particular, disconnected molecules or
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
        If test_size is also None, it will be set to 0.8.

    test_size : float, default=None
        The fraction of data to be used for the test subset. If None, it is set to 1 - train_size.
        If train_size is also None, it will be set to 0.2.

    include_chirality: bool, default=False
        Whether to take chirality of molecules into consideration.

    return_indices : bool, default=False
        Whether the method should return the input object subsets, i.e. SMILES strings
        or RDKit `Mol` objects, or only the indices of the subsets instead of the data.

    Returns
    ----------
    subsets : tuple[list, list, ...]
    Tuple with train-test subsets of provided arrays. First two are lists of SMILES strings or RDKit `Mol` objects,
    depending on the input type. If `return_indices` is True, only lists of indices are returned,
    and any additional data is ignored.

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
    if train_size is None and test_size is None:
        raise ValueError("Either train_size or test_size must be provided")
    if (
        train_size is not None
        and test_size is not None
        and not np.isclose(train_size + test_size, 1.0)
    ):
        raise ValueError("train_size and test_size must sum to 1.0")

    train_size, test_size = fill_missing_sizes(train_size, test_size)
    scaffolds = _create_scaffolds(list(data), include_chirality)
    scaffold_sets = sorted(scaffolds.values(), key=len)

    train_ids: list[int] = []
    test_ids: list[int] = []
    desired_test_size = int(test_size * len(data))

    for scaffold_set in scaffold_sets:
        if len(test_ids) < desired_test_size:
            test_ids.extend(scaffold_set)
        else:
            train_ids.extend(scaffold_set)

    if not train_ids:
        raise ValueError(
            "Train set is empty. Adjust the train_size or check provided data."
        )
    if not test_ids:
        raise ValueError(
            "Test set is empty. Adjust the test_size or check provided data."
        )

    train_subset: list[Any] = []
    test_subset: list[Any] = []

    if return_indices:
        train_subset = train_ids
        test_subset = test_ids
    else:
        train_subset = get_data_from_indices(data, train_ids)
        test_subset = get_data_from_indices(data, test_ids)

    ensure_nonempty_list(train_subset)
    ensure_nonempty_list(test_subset)

    additional_data_split: list[Sequence[Any]] = []

    additional_data_split = (
        split_additional_data(list(additional_data), [train_ids, test_ids])
        if additional_data
        else []
    )

    if additional_data:
        return train_subset, test_subset, *additional_data_split
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
    valid_size: Optional[float] = None,
    test_size: Optional[float] = None,
    include_chirality: bool = False,
    return_indices: bool = False,
) -> Union[
    tuple[
        Sequence[Union[str, Mol]],
        Sequence[Union[str, Mol]],
        Sequence[Union[str, Mol]],
        Sequence[Sequence[Any]],
    ],
    tuple[Sequence, ...],
    tuple[Sequence[int], Sequence[int], Sequence[int]],
]:
    """
    Split a list of SMILES or RDKit `Mol` objects into train and test subsets using Bemis-Murcko [1]_ scaffolds.

    This function ensures that similar molecules (sharing the same scaffold) are in the same split.
    This method, suggested in the MoleculeNet [2]_ paper, helps to test the model's ability to
    generalize to entirely new scaffolds. MoleculeNet introduced the scaffold split as an approximation
    to the time split, assuming that new molecules (test set) will be structurally different in terms of
    scaffolds from the training set.

    This approach is known to have certain limitations. In particular, disconnected molecules or
    molecules with no rings will not get a scaffold, resulting in them being grouped together
    regardless of their structure.

    The split is fully deterministic, with the smallest scaffold sets assigned to the test
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
        or RDKit `Mol` objects, or only the indices of the subsets instead of the data.

    Returns
    ----------
    subsets : tuple[list, list, ...]
    Tuple with train-test subsets of provided arrays. First two are lists of SMILES strings or RDKit `Mol` objects,
    depending on the input type. If `return_indices` is True, only lists of indices are returned,
    and any additional data is ignored.

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
    train_size, valid_size, test_size = validate_train_valid_test_split_sizes(
        train_size, valid_size, test_size
    )

    scaffolds = _create_scaffolds(list(data), include_chirality)
    scaffold_sets = sorted(scaffolds.values(), key=len)

    train_ids: list[int] = []
    valid_ids: list[int] = []
    test_ids: list[int] = []
    desired_test_size = int(test_size * len(data))
    desired_valid_size = int((test_size + valid_size) * len(data))

    for scaffold_set in scaffold_sets:
        if len(test_ids) < desired_test_size:
            test_ids.extend(scaffold_set)
        elif len(valid_ids) < desired_valid_size:
            valid_ids.extend(scaffold_set)
        else:
            train_ids.extend(scaffold_set)

    if not train_ids:
        raise ValueError(
            "Train set is empty. Adjust the train_size or check provided data."
        )
    if not test_ids:
        raise ValueError(
            "Test set is empty. Adjust the test_size or check provided data."
        )

    train_subset: list[Any] = []
    valid_subset: list[Any] = []
    test_subset: list[Any] = []

    if return_indices:
        train_subset = train_ids
        valid_subset = valid_ids
        test_subset = test_ids
    else:
        train_subset = get_data_from_indices(data, train_ids)
        valid_subset = get_data_from_indices(data, valid_ids)
        test_subset = get_data_from_indices(data, test_ids)

    ensure_nonempty_list(train_subset)
    ensure_nonempty_list(valid_subset)

    if len(valid_subset) == 0:
        warnings.warn(
            "Warning: Valid subset is empty. Consider using scaffold_train_test_split instead."
        )

    additional_data_split: list[Sequence[Any]] = []

    additional_data_split = (
        split_additional_data(list(additional_data), [train_ids, valid_ids, test_ids])
        if additional_data
        else []
    )

    if additional_data:
        return train_subset, valid_subset, test_subset, *additional_data_split
    else:
        return train_subset, valid_subset, test_subset


def _create_scaffolds(
    data: list[Union[str, Mol]], include_chirality: bool = False
) -> dict[str, list]:
    """
    Generate Bemis-Murcko scaffolds for a list of SMILES strings or RDKit `Mol` objects.
    This implementation uses Bemis-Murcko scaffolds to group molecules.
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
