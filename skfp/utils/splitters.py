import numbers
import warnings
from collections import defaultdict
from collections.abc import Sequence
from itertools import chain
from typing import Any, Union

import numpy as np
from rdkit.Chem import Mol
from rdkit.Chem.Scaffolds import MurckoScaffold

import sklearn.utils as skutils
from skfp.utils.validators import ensure_mols, Interval, RealNotInt, validate_params

@validate_params(
    {
        "data": ["sequence"],
        "additional_data": ["list"],
        "train_size": [
            Interval(RealNotInt, 0, 1, closed="neither"),
            Interval(numbers.Integral, 1, None, closed="left"),
            None,
        ],
        "test_size": [
            Interval(RealNotInt, 0, 1, closed="neither"),
            Interval(numbers.Integral, 1, None, closed="left"),
            None,
        ],
        "include_chirality": ["boolean"],
        "return_indices": ["boolean"],
    }
)
def scaffold_train_test_split(
    data: Sequence[Union[str, Mol]],
    *additional_data: list[Sequence],
    train_size: float = None,
    test_size: float = None,
    include_chirality: bool = False,    
    return_indices: bool = False,
) -> Union[
    tuple[list[str], list[str]],
    tuple[list[Mol], list[Mol]],
    tuple[list[int], list[int]],
]:
    """
    Split a list of SMILES or RDKit `Mol` objects into train and test subsets using Bemis-Murcko [1]_ scaffolds.

    This function ensures that similar molecules (sharing the same scaffold) are in the same split.
    This method, suggested in the MoleculeNet [2]_ paper, helps to test the model's ability to
    generalize to entirely new scaffolds. MoleculeNet introduced the scaffold split as an approximation 
    to the time split, assuming that new molecules (test set) will be structurally different in terms of 
    scaffolds from the training set.

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
    if train_size is None and test_size in None:
        raise ValueError("Either train_size or test_size must be provided")
    elif train_size == 0.0 and test_size == 0.0:
        return ValueError("Both train_size and test_size must be positive")
    elif train_size is None:
        train_size = 1 - test_size
    elif test_size is None:
        test_size = 1 - train_size  
    if not np.isclose(train_size + test_size , 1.0):
        raise ValueError("train_size and test_size must sum to 1.0")

    scaffolds = _create_scaffolds(list(data), include_chirality)
    scaffold_sets = sorted(scaffolds.values(), key=len)

    train_ids: list[int] = []
    test_ids: list[int] = []
    desired_test_size: int = int(test_size * len(data)) 

    for scaffold_set in scaffold_sets:
        if len(test_ids) < desired_test_size:
            test_ids.extend(scaffold_set)
        else:
            train_ids.extend(scaffold_set)

    if return_indices:
        train_subset = train_ids
        test_subset = test_ids
    else:
        train_subset = _get_data_from_indices(data, train_ids)
        test_subset = _get_data_from_indices(data, test_ids)

    if len(train_subset) == 0 or len(test_subset) == 0:
        raise ValueError("Either train or test subset is empty.")
    
    additional_data_split: list[Sequence[Any]] = []

    if additional_data:
        additional_data_split = list(
            chain.from_iterable(
                ((_safe_indexing(a, train_ids),
                _safe_indexing(a, test_ids))
                for a in additional_data)
            )
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
            Interval(numbers.Integral, 1, None, closed="left"),
            None,
        ],
        "valid_size": [
            Interval(RealNotInt, 0, 1, closed="neither"),
            Interval(numbers.Integral, 1, None, closed="left"),
            None,
        ],
        "test_size": [
            Interval(RealNotInt, 0, 1, closed="neither"),
            Interval(numbers.Integral, 1, None, closed="left"),
            None,
        ],
        "include_chirality": ["boolean"],
        "return_indices": ["boolean"],
    }
)
def scaffold_train_valid_test_split(
    data: Sequence[Union[str, Mol]],
    *additional_data: list[Sequence],
    train_size: float = None,
    test_size: float = None,
    valid_size: float = None,
    include_chirality: bool = False,
    return_indices: bool = False,
) -> Union[
    tuple[list[str], list[str], list[str]],
    tuple[list[Mol], list[Mol], list[Mol]],
    tuple[list[int], list[int], list[int]],
]:
    # TODO: Poprawić zwracany typ danych
    """
    Split a list of SMILES or RDKit `Mol` objects into train and test subsets using Bemis-Murcko [1]_ scaffolds.

    This function ensures that similar molecules (sharing the same scaffold) are in the same split.
    This method, suggested in the MoleculeNet [2]_ paper, helps to test the model's ability to
    generalize to entirely new scaffolds. MoleculeNet introduced the scaffold split as an approximation 
    to the time split, assuming that new molecules (test set) will be structurally different in terms of 
    scaffolds from the training set.

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
        The fraction of data to be used for the test subset. If None, it is set to 1 - train_size - valid_size. If train_size, test_size and  valid_size aren't set, train_size is set to 0.1. 

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
    if train_size is None and test_size is None:
        raise ValueError("Either train_size or test_size must be provided")
    if valid_size == 0.0:
        warnings.warn(
            "Validation set will not be returned since valid_size was set to 0.0."
            "Consider using train_test_split instead."
            )
    if train_size is None:
        train_size = 1 - test_size - valid_size
    if test_size is None:
        test_size = 1 - train_size - valid_size
    if valid_size is None:
        valid_size = 1 - test_size - train_size

    if not np.isclose(train_size + test_size + valid_size, 1.0):
        raise ValueError("train_size, test_size, and valid_size must sum to 1.0")

    scaffolds = _create_scaffolds(list(data), include_chirality)
    scaffold_sets = sorted(scaffolds.values(), key=len)

    train_ids: list[int] = []
    valid_ids: list[int] = []
    test_ids: list[int] = []

    desired_test_size: int = int(test_size * len(data))
    desired_valid_size: int = int((test_size + valid_size) * len(data))

    for scaffold_set in scaffold_sets:
        if len(test_ids) < desired_test_size:
            test_ids.extend(scaffold_set)
        elif len(valid_ids) < desired_valid_size:
            valid_ids.extend(scaffold_set)
        else:
            train_ids.extend(scaffold_set)

    if return_indices:
        train_subset = train_ids
        valid_subset = valid_ids
        test_subset = test_ids
    else:
        train_subset = _get_data_from_indices(data, train_ids)
        valid_subset = _get_data_from_indices(data, valid_ids)
        test_subset = _get_data_from_indices(data, test_ids)
    
    if len(train_subset) == 0 or len(test_subset) == 0:
        raise ValueError("Train or test subset is empty.")
    
    if len(valid_subset) == 0:
        warnings.warn("Warning: Valid subset is empty. Consider using scaffold_train_test_split instead.")

    additional_data_split: list[Sequence[Any]] = []

    if additional_data:
        additional_data_split = list(
            chain.from_iterable(
                ((_safe_indexing(a, train_ids),
                _safe_indexing(a, valid_ids),
                _safe_indexing(a, test_ids))
                for a in additional_data)
            )
        )

    if additional_data:
        return train_subset, valid_subset, test_subset, additional_data_split
    else:
        return train_subset, valid_subset, test_subset


def _create_scaffolds(
    data: list[Union[str, Mol]], include_chirality: bool = False
) -> dict[str, list]:
    """
    Generate Bemis-Murcko scaffolds for a list of SMILES strings or RDKit `Mol` objects.
    This implementation uses Bemis-Murcko scaffolds [1]_ to group molecules.
    Each scaffold is represented as a SMILES string.

    Parameters
    ----------
    data : sequence
        List of SMILES strings or RDKit `Mol` objects.

    include_chirality: bool, default=False
        Whether to take chirality of molecules into consideration.

    Returns
    -------
    scaffolds : dict
        A dictionary where keys are Bemis-Murcko scaffolds (as SMILES strings) and
        values are lists of indices pointing to molecules sharing the same scaffold.

    """
    scaffolds = defaultdict(list)
    molecules = ensure_mols(data)

    for ind, molecule in enumerate(molecules):
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=molecule, includeChirality=include_chirality
        )
        scaffolds[scaffold].append(ind)

    return scaffolds


def _get_data_from_indices(
    data: Sequence[Union[str, Mol]], indices: Sequence[int]
) -> list[Union[str, Mol]]:
    """
    Helper function to retrieve data from indices.
    """
    indices = set(indices)
    return [data[idx] for idx in indices]

