from collections import defaultdict
from collections.abc import Sequence
from typing import Union

import numpy as np
from rdkit.Chem import Mol
from rdkit.Chem.Scaffolds import MurckoScaffold

from skfp.utils.validators import ensure_mols


def scaffold_train_test_split(
    data: Sequence[Union[str, Mol]],
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
    Split a list of SMILES or RDKit 'Mol' objects into train and test subsets using Bemis-Murcko [1]_ scaffolds.

    This function ensures that similar molecules (sharing the same scaffold) are in the same split.
    This method, suggested in the MoleculeNet [2]_ paper, helps to test the model's ability to
    generalize to entirely new scaffolds.

    The split is fully deterministic, with the smallest scaffold sets assigned to the test
    subset and the largest to the training subset.

    The split fractions (train_size, test_size) must sum to 1.

    Parameters
    ----------
    data : sequence
        Sequence representing either SMILES strings or RDKit 'Mol' objects.

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
        or RDKit 'Mol' objects depending on the input
        - if `return_indices` is True, returns lists of indices

    References
    ----------
    .. [1] Bemis, G. W., & Murcko, M. A. (1996). The properties of known drugs. 1. Molecular frameworks.
    Journal of Medicinal Chemistry, 39(15), 2887-2893.
    .. [2] Z. Wu, B. Ramsundar, E. N. Feinberg, J. Gomes, C. Geniesse, A. S. Pappu, K. Leswing, V. Pande (2017).
    MoleculeNet: A Benchmark for Molecular Machine Learning. Chemical Science, 9(2), 513-530.
    """
    if train_size is None and test_size is None:
        raise ValueError("Either train_size or test_size must be provided.")
    if train_size is None:
        train_size = 1.0 - test_size
    elif test_size is None:
        test_size = 1.0 - train_size

    scaffolds = _create_scaffolds(list(data), include_chirality)
    scaffold_sets = sorted(scaffolds.values(), key=len)

    train_cutoff: int = int(train_size * len(data))
    train_ids: list[int] = []
    test_ids: list[int] = []

    for scaffold_set in scaffold_sets:
        if len(train_ids) < train_cutoff:
            train_ids.extend(scaffold_set)
        else:
            test_ids.extend(scaffold_set)

    if return_indices:
        train_subset = train_ids
        test_subset = test_ids
    else:
        train_subset = _get_data_from_indices(data, scaffolds, train_ids)
        test_subset = _get_data_from_indices(data, scaffolds, test_ids)

    return train_subset, test_subset


def scaffold_train_valid_test_split(
    data: Sequence[Union[str, Mol]],
    train_size: float = 0.8,
    test_size: float = 0.1,
    valid_size: float = 0.1,
    include_chirality: bool = False,
    return_indices: bool = False,
) -> Union[
    tuple[list[str], list[str], list[str]],
    tuple[list[Mol], list[Mol], list[Mol]],
    tuple[list[int], list[int], list[int]],
]:
    """
    Split a list of SMILES or RDKit 'Mol' objects into train, validation,
    and test subsets using Bemis-Murcko [1]_ scaffolds.

    This function ensures that similar molecules (sharing the same scaffold) are in the same split.
    This method, suggested in the MoleculeNet [2]_ paper, helps to test the model's ability to
    generalize to entirely new scaffolds.

    The split is fully deterministic, with the smallest scaffold sets assigned to the test
    subset and the largest to the training subset.

    The split fractions (train_size, test_size) must sum to 1.

    Parameters
    ----------
    data : sequence
        Sequence representing either SMILES strings or RDKit 'Mol' objects.

    train_size : float, default=0.8
        The fraction of data to be used for the train subset.

    test_size : float, default=0.1
        The fraction of data to be used for the validation subset.

    valid_size : float, default=0.1
        The fraction of data to be used for the test subset.

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
        or RDKit 'Mol' objects depending on the input
        - if `return_indices` is True, returns lists of indices

    References
    ----------
    .. [1] Bemis, G. W., & Murcko, M. A. (1996). The properties of known drugs. 1. Molecular frameworks.
    Journal of Medicinal Chemistry, 39(15), 2887-2893.
    .. [2] Z. Wu, B. Ramsundar, E. N. Feinberg, J. Gomes, C. Geniesse, A. S. Pappu, K. Leswing, V. Pande (2017).
    MoleculeNet: A Benchmark for Molecular Machine Learning. Chemical Science, 9(2), 513-530.
    """
    np.testing.assert_almost_equal(train_size + test_size + valid_size, 1.0)

    scaffolds = _create_scaffolds(list(data), include_chirality)
    scaffold_sets = sorted(scaffolds.values(), key=len)

    train_cutoff: int = int(train_size * len(data))
    valid_cutoff: int = int((train_size + valid_size) * len(data))

    train_ids: list[int] = []
    valid_ids: list[int] = []
    test_ids: list[int] = []

    for scaffold_set in scaffold_sets:
        if len(train_ids) < train_cutoff:
            train_ids.extend(scaffold_set)
        elif len(valid_ids) < valid_cutoff:
            valid_ids.extend(scaffold_set)
        else:
            test_ids.extend(scaffold_set)

    if return_indices:
        train_subset = train_ids
        valid_subset = valid_ids
        test_subset = test_ids
    else:
        train_subset = _get_data_from_indices(data, scaffolds, train_ids)
        valid_subset = _get_data_from_indices(data, scaffolds, valid_ids)
        test_subset = _get_data_from_indices(data, scaffolds, test_ids)
    
    return train_subset, valid_subset, test_subset


def _create_scaffolds(
    data: list[Union[str, Mol]], include_chirality: bool = False
) -> dict[str, list]:
    """
    Generate Bemis-Murcko scaffolds for a list of SMILES strings or RDKit 'Mol' objects.

    Parameters
    ----------
    data : sequence
        List of SMILES strings or RDKit 'Mol' objects.

    include_chirality: bool, default=False
        Whether to take chirality of molecules into consideration.

    Returns
    -------
    scaffolds : dict
        A dictionary where keys are Bemis-Murcko scaffolds (as SMILES strings) and
        values are lists of indices pointing to molecules sharing the same scaffold.

    Notes
    -----
    - This implementation uses Bemis-Murcko scaffolds (Bemis and Murcko, 1996) to group molecules.
    - Each scaffold is represented as a SMILES string.
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
    data: Sequence[Union[str, Mol]], scaffolds: dict[str, list], indices: Sequence[int]
) -> list[Union[str, Mol]]:
    """
    Helper function to retrieve data from indices.
    """
    result = []
    index_set = set(indices)

    for i in index_set:
        if isinstance(data[i], str):
            item = data[i]
        elif isinstance(data[i], Mol):
            item = data[i]
        else:
            raise ValueError("Unsupported data type in input.")

        result.append(item)

    return result
