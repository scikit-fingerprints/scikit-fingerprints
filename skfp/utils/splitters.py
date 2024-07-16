from collections import defaultdict
from collections.abc import Sequence
from typing import Union

import numpy as np
from rdkit.Chem import Mol
from rdkit.Chem.Scaffolds import MurckoScaffold

from skfp.utils.validators import ensure_mols


def create_scaffolds(
    data: list[Union[str, Mol]], include_chirality: bool = False
) -> dict:
    """
    Generate Bemis-Murcko scaffolds for a list of SMILES strings or RDKit Mol objects.

    Parameters
    ----------
    data : list[str] or list[Mol]
        List of SMILES strings or RDKit Mol objects.

    include_chirality: bool, default=False
        Should the method take chirality of molecules into consideration.

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


def scaffold_train_test_split(
    data: Sequence[Union[str, Mol]],
    train_size: float = 0.8,
    test_size: float = 0.2,
    return_indices: bool = False,
    include_chirality: bool = False,
) -> tuple[list, list]:
    """
    Split a list of SMILES or RDKit Mol objects into train and test subsets using Bemis-Murcko scaffolds.

    This function ensures that similar molecules (sharing the same scaffold) are in the same split.
    This method, suggested in the MoleculeNet paper, helps to test the model's ability to
    generalize to entirely new scaffolds.

    Parameters
    ----------
    data : list[str]
        Sequence representing either SMILES strings or RDKit Mol objects.

    train_size : float, default=0.8
        The fraction of data to be used for the training split.

    test_size : float, default=0.2
        The fraction of data to be used for the test split.

    return_indices : bool, default=False
        Should the method return SMILES/RDKit Mol objects or just indices in the dataset.

    include_chirality: bool, default=False
        Should the method take chirality of molecules into consideration.

    Returns
    ----------
    subsets : tuple[list, list]
        A tuple of train and test subsets. The format depends on `return_indices`:
        - If `return_indices` is False, returns lists of SMILES strings or RDKit Mol objects depending
        on the input.
        - If `return_indices` is True, returns lists of indices.

    Notes
    -----
    - This implementation uses Bemis-Murcko scaffolds (Bemis and Murcko, 1996) to group molecules.
    - The split is fully deterministic, with the smallest scaffold sets assigned to the test
    subset and the largest to the training subset.
    - The split fractions (train_size, test_size) must sum to 1.

    References
    ----------
    - Bemis, G. W., & Murcko, M. A. (1996). The properties of known drugs. 1. Molecular frameworks.
    Journal of Medicinal Chemistry, 39(15), 2887-2893.
    - Z. Wu, B. Ramsundar, E. N. Feinberg, J. Gomes, C. Geniesse, A. S. Pappu, K. Leswing, V. Pande (2017).
    MoleculeNet: A Benchmark for Molecular Machine Learning. Chemical Science, 9(2), 513-530.
    """
    np.testing.assert_almost_equal(train_size + test_size, 1.0)

    scaffolds = create_scaffolds(list(data), include_chirality)
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
        return train_ids, test_ids
    else:
        return _get_data_from_indices(
            data, scaffolds, train_ids
        ), _get_data_from_indices(data, scaffolds, test_ids)


def scaffold_train_valid_test_split(
    data: Sequence[Union[str, Mol]],
    train_size: float = 0.8,
    test_size: float = 0.1,
    valid_size: float = 0.1,
    return_indices: bool = False,
    include_chirality: bool = False,
) -> tuple[list, list, list]:
    """
    Split a list of SMILES or RDKit Mol objects into train, validation,
    and test subsets using Bemis-Murcko scaffolds.

    This function ensures that similar molecules (sharing the same scaffold) are in the same split.
    This method, suggested in the MoleculeNet paper, helps to test the model's ability to generalize
    to entirely new scaffolds.

    Parameters
    ----------
    data : list[str]
        Sequence representing either SMILES strings or RDKit Mol objects.

    frac_train : float, default=0.8
        The fraction of data to be used for the training split.

    frac_valid : float, default=0.1
        The fraction of data to be used for the validation split.

    frac_test : float, default=0.1
        The fraction of data to be used for the test split.

    return_indices : bool, default=False
        Should the method return SMILES/RDKit Mol objects or just indices in the dataset.

    include_chirality: bool, default=False
        Should the method take chirality of molecules into consideration?


    Returns
    ----------
    subsets : tuple[list, list, list]
        A tuple of train, validation, and test subsets. The format depends on `return_indices`:
        - If `return_indices` is False, returns lists of SMILES strings
        or RDKit Mol objects depending on the input.
        - If `return_indices` is True, returns lists of indices.

    Notes
    -----
    - This implementation uses Bemis-Murcko scaffolds (Bemis and Murcko, 1996) to group molecules.
    - The split is fully deterministic, with the smallest scaffold sets assigned to the test subset,
    larger sets to the validation subset,
      and the largest sets to the training subset.
    - The test and validation sets can be slightly larger than specified,
    ensuring they are at least the desired fraction, but as close as possible in size.
    - The split fractions (frac_train, frac_valid, frac_test) must sum to 1.

    References
    ----------
    - Bemis, G. W., & Murcko, M. A. (1996). The properties of known drugs. 1. Molecular frameworks.
    Journal of Medicinal Chemistry, 39(15), 2887-2893.
    - Z. Wu, B. Ramsundar, E. N. Feinberg, J. Gomes, C. Geniesse, A. S. Pappu, K. Leswing, V. Pande (2017).
    MoleculeNet: A Benchmark for Molecular Machine Learning. Chemical Science, 9(2), 513-530.
    """
    np.testing.assert_almost_equal(train_size + test_size + valid_size, 1.0)

    scaffolds = create_scaffolds(list(data), include_chirality)
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
        return train_ids, valid_ids, test_ids
    else:
        return (
            _get_data_from_indices(data, scaffolds, train_ids),
            _get_data_from_indices(data, scaffolds, valid_ids),
            _get_data_from_indices(data, scaffolds, test_ids),
        )


def _get_data_from_indices(data, scaffolds, indices):
    """
    Helper function to retrieve data from indices.
    """
    return [data[i] if isinstance(data[0], str) else scaffolds[i] for i in indices]
