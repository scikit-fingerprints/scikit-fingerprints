import warnings
from collections.abc import Sequence
from numbers import Integral
from typing import Any, Optional, Union

from rdkit import DataStructs
from rdkit.Chem import AllChem, Mol
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
from sklearn.utils._param_validation import Interval, RealNotInt, validate_params

from skfp.model_selection.splitters.utils import (
    ensure_nonempty_subset,
    get_data_from_indices,
    split_additional_data,
    validate_train_test_split_sizes,
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
        "return_indices": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def maxmin_train_test_split(
    data: Sequence[Union[str, Mol]],
    *additional_data: Sequence,
    train_size: Optional[float] = None,
    test_size: Optional[float] = None,
    return_indices: bool = False,
    random_state: int = 0,
) -> Union[
    tuple[
        Sequence[Union[str, Mol]], Sequence[Union[str, Mol]], Sequence[Sequence[Any]]
    ],
    tuple[Sequence, ...],
    tuple[Sequence[int], Sequence[int]],
]:
    """
    Split using maxmin algorithm or picking a subset of item from a pool.
    Implements specific

    Parameters
    ----------
    data : sequence
        A sequence representing either SMILES strings or RDKit `Mol` objects.

    additional_data: list[sequence]
        Additional sequences to be split alongside the main data (e.g., labels or feature vectors).

    train_size : float, default=None
        The fraction of data to be used for the train subset. If None, it is set to 1 - test_size.
        If test_size is also None, it will be set to 0.8.

    test_size : float, default=None
        The fraction of data to be used for the test subset. If None, it is set to 1 - train_size.
        If train_size is also None, it will be set to 0.2.

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
    https://github.com/deepchem/deepchem
    https://rdkit.org/docs/cppapi/classRDPickers_1_1MaxMinPicker.html
    https://squonk.it/docs/cells/RDKit%20MaxMin%20Picker/
    https://rdkit.blogspot.com/2017/11/revisting-maxminpicker.html

    """
    data_size = len(data)
    train_size, test_size = validate_train_test_split_sizes(
        train_size, test_size, data_size
    )

    molecules = ensure_mols(data)
    fingerprints = [
        AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024) for mol in molecules
    ]

    picker = MaxMinPicker()
    test_idxs = picker.LazyPick(
        distFunc=lambda i, j: 1
        - DataStructs.DiceSimilarity(fingerprints[i], fingerprints[j]),
        poolSize=data_size,
        pickSize=test_size,
        seed=random_state,
    )

    train_idxs = list(set(range(data_size)) - set(test_idxs))
    train_subset: list[Any] = []
    test_subset: list[Any] = []

    if return_indices:
        train_subset = train_idxs
        test_subset = test_idxs
    else:
        train_subset = get_data_from_indices(data, train_idxs)
        test_subset = get_data_from_indices(data, test_idxs)

    ensure_nonempty_subset(train_subset, "train")
    ensure_nonempty_subset(test_subset, "test")

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
        "test_size": [
            Interval(RealNotInt, 0, 1, closed="neither"),
            Interval(Integral, 1, None, closed="left"),
            None,
        ],
        "include_chirality": ["boolean"],
        "return_indices": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def maxmin_train_valid_test_split(
    data: Sequence[Union[str, Mol]],
    *additional_data: Sequence,
    train_size: Optional[float] = None,
    valid_size: Optional[float] = None,
    test_size: Optional[float] = None,
    return_indices: bool = False,
    random_state: int = 0,
) -> Union[
    tuple[
        Sequence[Union[str, Mol]], Sequence[Union[str, Mol]], Sequence[Sequence[Any]]
    ],
    tuple[Sequence, ...],
    tuple[Sequence[int], Sequence[int]],
]:

    data_size = len(data)
    train_size, valid_size, test_size = validate_train_valid_test_split_sizes(
        train_size, valid_size, test_size, len(data)
    )
    molecules = ensure_mols(data)
    fingerprints = [
        AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024) for mol in molecules
    ]
    picker = MaxMinPicker()

    test_idxs = picker.LazyPick(
        distFunc=lambda i, j: 1
        - DataStructs.DiceSimilarity(fingerprints[i], fingerprints[j]),
        poolSize=data_size,
        pickSize=test_size,
        seed=random_state,
    )

    valid_idxs = picker.LazyPick(
        distFunc=lambda i, j: 1
        - DataStructs.DiceSimilarity(fingerprints[i], fingerprints[j]),
        poolSize=data_size,
        pickSize=test_size + valid_size,
        firstPicks=test_idxs,
        seed=random_state,
    )

    train_idxs = list(set(range(data_size)) - set(test_idxs) - set(valid_idxs))
    valid_idxs = list(set(valid_idxs) - set(test_idxs))
    train_subset: list[Any] = []
    valid_subset: list[Any] = []
    test_subset: list[Any] = []

    if return_indices:
        train_subset = train_idxs
        test_subset = test_idxs
        valid_subset = valid_idxs
    else:
        train_subset = get_data_from_indices(data, train_idxs)
        test_subset = get_data_from_indices(data, test_idxs)
        valid_subset = get_data_from_indices(data, valid_idxs)

    ensure_nonempty_subset(train_subset, "train")
    ensure_nonempty_subset(test_subset, "test")

    if len(valid_subset) == 0:
        warnings.warn(
            "Warning: Valid subset is empty. Consider using maxmin_train_test_split instead."
        )

    if additional_data:
        additional_data_split: list[Sequence[Any]] = split_additional_data(
            list(additional_data), train_idxs, valid_idxs, test_idxs
        )
        return train_subset, valid_subset, test_subset, additional_data_split
    else:
        return train_subset, valid_subset, test_subset
