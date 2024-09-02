import warnings
from collections import defaultdict
from collections.abc import Sequence
from numbers import Integral
from typing import Any, Optional, Union

from rdkit.Chem import Mol
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.utils._param_validation import Interval, RealNotInt, validate_params

from skfp.model_selection.utils import (
    ensure_nonempty_subset,
    get_data_from_indices,
    split_additional_data,
    validate_train_test_sizes,
    validate_train_valid_test_split_sizes,
)
from skfp.utils.validators import ensure_mols


@validate_params(
    {
        "data": ["array-like"],
        "additional_data": ["tuple"],
        "train_size": [
            Interval(RealNotInt, 0, 1, closed="neither"),
            Interval(Integral, 1, None, closed="left"),
            None,
        ],
        "test_size": [
            Interval(RealNotInt, 0, 1, closed="neither"),
            Interval(Integral, 1, None, closed="left"),
            None,
        ],
        "include_chirality": ["boolean"],
        "use_csk": ["boolean"],
        "return_indices": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def scaffold_train_test_split(
    data: Sequence[Union[str, Mol]],
    *additional_data: Sequence,
    train_size: Optional[float] = None,
    test_size: Optional[float] = None,
    include_chirality: bool = False,
    use_csk: bool = False,
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

    The `use_csk` parameter allows to choose between using the core structure scaffold (which includes atom types)
    and the skeleton scaffold (which does not) [3]_. This functionality only works correctly for molecules where
    all atoms have a degree of 4 or less. Molecules with atoms having a degree greater than 4 raise an error because
    core structure scaffolds (CSKs) with carbons can't handle these cases properly.

    This approach is known to have certain limitations. In particular, molecules with no rings will not get a scaffold,
    resulting in them being grouped together regardless of their structure.

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

    use_csk: bool, default=False
        Whether to use molecule's skeleton or the core structure scaffold (including atom types).

    return_indices : bool, default=False
        Whether the method should return the input object subsets, i.e. SMILES strings
        or RDKit `Mol` objects, or only the indices of the subsets instead of the data.

    Returns
    ----------
    subsets : tuple[list, list, ...]
    Tuple with train-test subsets of provided arrays. First two are lists of SMILES strings or RDKit `Mol` objects,
    depending on the input type. If `return_indices` is True, lists of indices are returned instead of actual data.
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

    .. [3] ` Bemis-Murcko scaffolds and their variants
        https://github.com/rdkit/rdkit/discussions/6844` _


    """
    train_size, test_size = validate_train_test_sizes(train_size, test_size)
    scaffolds = _create_scaffolds(data, include_chirality, use_csk)
    scaffold_sets = sorted(scaffolds.values(), key=len)

    train_idxs: list[int] = []
    test_idxs: list[int] = []
    desired_test_size = max(1, int(test_size * len(data))) if test_size > 0 else 0

    for scaffold_set in scaffold_sets:
        if len(test_idxs) < desired_test_size:
            test_idxs.extend(scaffold_set)
        else:
            train_idxs.extend(scaffold_set)

    ensure_nonempty_subset(train_idxs, "Train")
    ensure_nonempty_subset(test_idxs, "Test")

    train_subset: list[Any] = []
    test_subset: list[Any] = []

    if return_indices:
        train_subset = train_idxs
        test_subset = test_idxs
    else:
        train_subset = get_data_from_indices(data, train_idxs)
        test_subset = get_data_from_indices(data, test_idxs)

    ensure_nonempty_subset(train_subset, "Train")
    ensure_nonempty_subset(test_subset, "Test")

    if additional_data:
        additional_data_split: list[Sequence[Any]] = split_additional_data(
            list(additional_data), train_idxs, test_idxs
        )
        return train_subset, test_subset, additional_data_split
    else:
        return train_subset, test_subset


@validate_params(
    {
        "data": ["array-like"],
        "additional_data": ["tuple"],
        "train_size": [
            Interval(RealNotInt, 0, 1, closed="neither"),
            Interval(Integral, 1, None, closed="left"),
            None,
        ],
        "valid_size": [
            Interval(RealNotInt, 0, 1, closed="neither"),
            Interval(Integral, 1, None, closed="left"),
            None,
        ],
        "test_size": [
            Interval(RealNotInt, 0, 1, closed="neither"),
            Interval(Integral, 1, None, closed="left"),
            None,
        ],
        "include_chirality": ["boolean"],
        "use_csk": ["boolean"],
        "return_indices": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def scaffold_train_valid_test_split(
    data: Sequence[Union[str, Mol]],
    *additional_data: Sequence,
    train_size: Optional[float] = None,
    valid_size: Optional[float] = None,
    test_size: Optional[float] = None,
    include_chirality: bool = False,
    use_csk: bool = False,
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

    The `use_csk` parameter allows to choose between using the core structure scaffold (which includes atom types)
    and the skeleton scaffold (which does not) [3]_. This functionality only works correctly for molecules where
    all atoms have a degree of 4 or less. Molecules with atoms having a degree greater than 4 raise an error because
    core structure scaffolds (CSKs) with carbons can't handle these cases properly.

    This approach is known to have certain limitations. In particular, molecules with no rings will not get a scaffold,
    resulting in them being grouped together regardless of their structure.

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

    valid_size : float, default=None
        The fraction of data to be used for the test subset. If None, it is set to 1 - train_size - valid_size.
        If train_size, test_size and valid_size aren't set, train_size is set to 0.1.

    test_size : float, default=None
        The fraction of data to be used for the validation subset. If None, it is set to 1 - train_size - valid_size.
        If valid_size is not provided, test_size is set to 1 - train_size. If train_size, test_size and
        valid_size aren't set, test_size is set to 0.1.

    include_chirality: bool, default=False
        Whether to take chirality of molecules into consideration.

    use_csk: bool, default=False
        Whether to use molecule's skeleton or the core structure scaffold (including atom types).

    return_indices : bool, default=False
        Whether the method should return the input object subsets, i.e. SMILES strings
        or RDKit `Mol` objects, or only the indices of the subsets instead of the data.

    Returns
    ----------
    subsets : tuple[list, list, ...]
    Tuple with train-test subsets of provided arrays. First two are lists of SMILES strings or RDKit `Mol` objects,
    depending on the input type. If `return_indices` is True, lists of indices are returned instead of actual data.

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

    .. [3] ` Bemis-Murcko scaffolds and their variants
        https://github.com/rdkit/rdkit/discussions/6844` _

    """
    train_size, valid_size, test_size = validate_train_valid_test_split_sizes(
        train_size, valid_size, test_size
    )

    scaffolds = _create_scaffolds(data, include_chirality, use_csk)
    scaffold_sets = sorted(scaffolds.values(), key=len)

    train_idxs: list[int] = []
    valid_idxs: list[int] = []
    test_idxs: list[int] = []
    desired_test_size = max(1, int(test_size * len(data))) if test_size > 0 else 0
    desired_valid_size = max(1, int(valid_size * len(data))) if valid_size > 0 else 0

    for scaffold_set in scaffold_sets:
        if len(test_idxs) < desired_test_size:
            test_idxs.extend(scaffold_set)
        elif len(valid_idxs) < desired_valid_size:
            valid_idxs.extend(scaffold_set)
        else:
            train_idxs.extend(scaffold_set)

    train_subset: list[Any] = []
    valid_subset: list[Any] = []
    test_subset: list[Any] = []

    if return_indices:
        train_subset = train_idxs
        valid_subset = valid_idxs
        test_subset = test_idxs
    else:
        train_subset = get_data_from_indices(data, train_idxs)
        valid_subset = get_data_from_indices(data, valid_idxs)
        test_subset = get_data_from_indices(data, test_idxs)

    ensure_nonempty_subset(train_subset, "Train")
    ensure_nonempty_subset(valid_subset, "Validation")
    ensure_nonempty_subset(test_subset, "Test")

    if additional_data:
        additional_data_split: list[Sequence[Any]] = split_additional_data(
            list(additional_data), train_idxs, valid_idxs, test_idxs
        )
        return train_subset, valid_subset, test_subset, additional_data_split
    else:
        return train_subset, valid_subset, test_subset


def _create_scaffolds(
    data: Sequence[Union[str, Mol]],
    include_chirality: bool = False,
    use_csk: bool = False,
) -> dict[str, list]:
    """
    Generate Bemis-Murcko scaffolds for a list of SMILES strings or RDKit `Mol` objects.
    This function groups molecules by their Bemis-Murcko scaffold, which can be generated
    as either the core structure scaffold (with atom types) or the skeleton scaffold
    (without atom types). Scaffolds can optionally include chirality information.
    """
    scaffolds = defaultdict(list)
    molecules = ensure_mols(data)

    for idx, mol in enumerate(molecules):
        if use_csk:
            scaffold = MurckoScaffold.MakeScaffoldGeneric(mol=mol)
        else:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol, includeChirality=include_chirality
            )

        scaffolds[scaffold].append(idx)

    return scaffolds
