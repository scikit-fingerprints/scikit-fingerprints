from collections import defaultdict
from collections.abc import Sequence
from copy import deepcopy
from numbers import Integral
from typing import Any, Optional, Union

from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.utils._param_validation import Interval, RealNotInt, validate_params

from skfp.model_selection.splitters.utils import (
    ensure_nonempty_subset,
    split_additional_data,
    validate_train_test_split_sizes,
    validate_train_valid_test_split_sizes,
)
from skfp.utils.functions import get_data_from_indices
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
    Split using groups of Bemis-Murcko scaffolds.

    This split uses deterministically partitioned groups of Bemis-Murcko molecular
    scaffolds [1]_ for splitting, as introduced in the MoleculeNet [2]_ paper. It aims
    to verify the model generalization to new and rare scaffolds, as an approximation
    to the time split.

    By default, core structure scaffolds are used (following RDKit), which include atom
    types. Original Bemis-Murcko approach uses the cyclic skeleton of a molecule, replacing
    all atoms by carbons. It is also known as CSK (Cyclic SKeleton) [3]_, and can be
    used with `use_csk` parameter.

    This approach is known to have certain limitations. In particular, molecules with
    no rings will not get a scaffold, resulting in them being grouped together regardless
    of their structure.

    The split is fully deterministic, with the smallest scaffold sets assigned to the test
    subset and the rest to the training subset.

    If ``train_size`` and ``test_size`` are integers, they must sum up to the ``data``
    length. If they are floating numbers, they must sum up to 1.

    Parameters
    ----------
    data : sequence
        A sequence representing either SMILES strings or RDKit ``Mol`` objects.

    additional_data: list[sequence]
        Additional sequences to be split alongside the main data (e.g., labels or feature vectors).

    train_size : float, default=None
        The fraction of data to be used for the train subset. If None, it is set to 1 - test_size.
        If test_size is also None, it will be set to 0.8.

    test_size : float, default=None
        The fraction of data to be used for the test subset. If None, it is set to 1 - train_size.
        If train_size is also None, it will be set to 0.2.

    use_csk: bool, default=False
        Whether to use molecule's skeleton or the core structure scaffold (including atom types).

    return_indices : bool, default=False
        Whether the method should return the input object subsets, i.e. SMILES strings
        or RDKit ``Mol`` objects, or only the indices of the subsets instead of the data.

    Returns
    -------
    subsets : tuple[list, list, ...]
        Tuple with train-test subsets of provided arrays. First two are lists of SMILES
        strings or RDKit ``Mol`` objects, depending on the input type. If `return_indices`
        is True, lists of indices are returned instead of actual data.

    References
    ----------
    .. [1] `Bemis, G. W., & Murcko, M. A.
        "The properties of known drugs. 1. Molecular frameworks."
        Journal of Medicinal Chemistry, 39(15), 2887-2893.
        <https://www.researchgate.net/publication/14493474_The_Properties_of_Known_Drugs_1_Molecular_Frameworks>`_

    .. [2] `Z. Wu, B. Ramsundar, E. N. Feinberg, J. Gomes, C. Geniesse, A. S. Pappu, K. Leswing, V. Pande
        "MoleculeNet: A Benchmark for Molecular Machine Learning."
        Chemical Science, 9(2), 513-530.
        <https://www.researchgate.net/publication/314182452_MoleculeNet_A_Benchmark_for_Molecular_Machine_Learning>`_

    .. [3] ` Bemis-Murcko scaffolds and their variants
        <https://github.com/rdkit/rdkit/discussions/6844>`_
    """
    train_size, test_size = validate_train_test_split_sizes(
        train_size, test_size, len(data)
    )

    scaffold_sets = _create_scaffold_sets(data, use_csk)
    scaffold_sets.sort(key=len)

    train_idxs: list[int] = []
    test_idxs: list[int] = []

    for scaffold_set in scaffold_sets:
        if len(test_idxs) < test_size:
            test_idxs.extend(scaffold_set)
        else:
            train_idxs.extend(scaffold_set)

    ensure_nonempty_subset(train_idxs, "train")
    ensure_nonempty_subset(test_idxs, "test")

    if return_indices:
        train_subset = train_idxs
        test_subset = test_idxs
    else:
        train_subset = get_data_from_indices(data, train_idxs)
        test_subset = get_data_from_indices(data, test_idxs)

    if additional_data:
        additional_data_split: list[Sequence[Any]] = split_additional_data(
            list(additional_data), train_idxs, test_idxs
        )
        return train_subset, test_subset, *additional_data_split
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
    Split using groups of Bemis-Murcko scaffolds.

    This split uses deterministically partitioned groups of Bemis-Murcko molecular
    scaffolds [1]_ for splitting, as introduced in the MoleculeNet [2]_ paper. It aims
    to verify the model generalization to new and rare scaffolds, as an approximation
    to the time split.

    By default, core structure scaffolds are used (following RDKit), which include atom
    types. Original Bemis-Murcko approach uses the cyclic skeleton of a molecule, replacing
    all atoms by carbons. It is also known as CSK (Cyclic SKeleton) [3]_, and can be
    used with `use_csk` parameter.

    This approach is known to have certain limitations. In particular, molecules with
    no rings will not get a scaffold, resulting in them being grouped together regardless
    of their structure.

    The split is fully deterministic, with the smallest scaffold sets assigned to the test
    subset, larger to the validation subset, and the rest to the training subset.

    If ``train_size``, ``valid_size`` and ``test_size`` are integers, they must sum up
    to the ``data`` length. If they are floating numbers, they must sum up to 1.

    Parameters
    ----------
    data : sequence
        A sequence representing either SMILES strings or RDKit ``Mol`` objects.

    additional_data: sequence
        Additional sequences to be split alongside the main data, e.g. labels.

    train_size : float, default=None
        The fraction of data to be used for the train subset. If None, it is set
        to 1 - test_size - valid_size. If valid_size is not provided, train_size
        is set to 1 - test_size. If train_size, test_size and valid_size aren't
        set, train_size is set to 0.8.

    valid_size : float, default=None
        The fraction of data to be used for the test subset. If None, it is set
        to 1 - train_size - valid_size. If train_size, test_size and valid_size
        aren't set, train_size is set to 0.1.

    test_size : float, default=None
        The fraction of data to be used for the validation subset. If None, it is
        set to 1 - train_size - valid_size. If valid_size is not provided, test_size
        is set to 1 - train_size. If train_size, test_size and valid_size aren't set,
        test_size is set to 0.1.

    use_csk: bool, default=False
        Whether to use the molecule cyclic skeleton (CSK), instead of the core
        structure scaffold.

    return_indices : bool, default=False
        Whether the method should return the input object subsets, i.e. SMILES strings
        or RDKit ``Mol`` objects, or only the indices of the subsets instead of the data.

    Returns
    -------
    subsets : tuple[list, list, ...]
        Tuple with train-valid-test subsets of provided arrays. First three are lists of
        SMILES strings or RDKit ``Mol`` objects, depending on the input type. If
        `return_indices` is True, lists of indices are returned instead of actual data.

    References
    ----------
    .. [1] `Bemis, G. W., & Murcko, M. A.
        "The properties of known drugs. 1. Molecular frameworks."
        Journal of Medicinal Chemistry, 39(15), 2887-2893.
        <https://www.researchgate.net/publication/14493474_The_Properties_of_Known_Drugs_1_Molecular_Frameworks>`_

    .. [2] `Z. Wu, B. Ramsundar, E. N. Feinberg, J. Gomes, C. Geniesse, A. S. Pappu, K. Leswing, V. Pande
        "MoleculeNet: A Benchmark for Molecular Machine Learning."
        Chemical Science, 9(2), 513-530.
        <https://www.researchgate.net/publication/314182452_MoleculeNet_A_Benchmark_for_Molecular_Machine_Learning>`_

    .. [3] ` Bemis-Murcko scaffolds and their variants
        <https://github.com/rdkit/rdkit/discussions/6844>`_
    """
    train_size, valid_size, test_size = validate_train_valid_test_split_sizes(
        train_size, valid_size, test_size, len(data)
    )

    scaffold_sets = _create_scaffold_sets(data, use_csk)
    scaffold_sets.sort(key=len)

    train_idxs: list[int] = []
    valid_idxs: list[int] = []
    test_idxs: list[int] = []

    for scaffold_set in scaffold_sets:
        if len(test_idxs) < test_size:
            test_idxs.extend(scaffold_set)
        elif len(valid_idxs) < valid_size:
            valid_idxs.extend(scaffold_set)
        else:
            train_idxs.extend(scaffold_set)

    ensure_nonempty_subset(train_idxs, "train")
    ensure_nonempty_subset(valid_idxs, "validation")
    ensure_nonempty_subset(test_idxs, "test")

    if return_indices:
        train_subset = train_idxs
        valid_subset = valid_idxs
        test_subset = test_idxs
    else:
        train_subset = get_data_from_indices(data, train_idxs)
        valid_subset = get_data_from_indices(data, valid_idxs)
        test_subset = get_data_from_indices(data, test_idxs)

    if additional_data:
        additional_data_split: list[Sequence[Any]] = split_additional_data(
            list(additional_data), train_idxs, valid_idxs, test_idxs
        )
        return train_subset, valid_subset, test_subset, *additional_data_split
    else:
        return train_subset, valid_subset, test_subset


def _create_scaffold_sets(
    data: Sequence[Union[str, Mol]], use_csk: bool = False
) -> list[list[int]]:
    """
    Generate Bemis-Murcko scaffolds for a list of SMILES strings or RDKit ``Mol`` objects.
    This function groups molecules by their Bemis-Murcko scaffold into sets of molecules
    with the same scaffold.

    They can be generated as either the core structure scaffold (with atom types) or the
    skeleton scaffold (without atom types).
    """
    scaffold_sets = defaultdict(list)
    mols = ensure_mols(data)

    for idx, mol in enumerate(mols):
        mol = deepcopy(mol)
        Chem.RemoveStereochemistry(mol)

        if use_csk:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
            scaffold = MurckoScaffold.GetScaffoldForMol(scaffold)
        else:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)

        scaffold_sets[scaffold].append(idx)

    scaffold_sets = list(scaffold_sets.values())
    return scaffold_sets
