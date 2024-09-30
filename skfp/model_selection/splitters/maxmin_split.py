import warnings
from collections.abc import Sequence
from numbers import Integral
from typing import Any, Optional, Union

from rdkit.Chem import Mol
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
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
    Split using MaxMin algorithm.

    MaxMinPicker is an efficient algorithm for picking a optimal subset of diverse compounds from a candidate pool.
    The algorithm is described in Ashton, M. [1]_,implementation was created by Roger Sayle [3]_ [4]_ [5]_
    at the 2017 RDKit user group meeting [2]_.[4]_

    It starts by vectorizing molecules with ECFP4 fingerprint. First test molecule is picked randomly. Each next one
    is selected to maximize the minimal distance to the already selected molecules (hence the MaxMin name) [4]_.,
    calculating and recording the distances as required. This molecule is the most distant
    one to those already picked so is transferred to the picked set [3]_.

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
    .. [1] `Mark Ashton, John Barnardb, Florence Casset, Michael Charlton, Geoffrey Downsb,
        Dominique Gorse, JohnHolliday, Roger Lahanac, Peter Willett
        "Identification of Diverse Database Subsets using Property-Based and Fragment-Based Molecular Descriptions"
        <https://onlinelibrary.wiley.com/doi/10.1002/qsar.200290002>_`
    .. [2] `Roger Sayle
        "Improved RDKit implementation"
        <https://github.com/rdkit/UGM_2017/blob/master/Presentations/Sayle_RDKitDiversity_Berlin17.pdf>_`
    .. [3] `Tim Dudgeon
        "Revisting the MaxMinPicker"
        <https://rdkit.org/docs/cppapi/classRDPickers_1_1MaxMinPicker.html>_`
    .. [4] `RDKit MaxMin Picker
        <https://squonk.it/docs/cells/RDKit%20MaxMin%20Picker>_`
    """
    data_size = len(data)
    train_size, test_size = validate_train_test_split_sizes(
        train_size, test_size, data_size
    )

    mols = ensure_mols(data)
    fps = GetMorganGenerator().GetFingerprints(mols)

    picker = MaxMinPicker()
    test_idxs = picker.LazyBitVectorPick(
        fps,
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
    """
    Split using MaxMin algorithm.

    MaxMinPicker is an efficient algorithm for picking a optimal subset of diverse compounds from a candidate pool.
    The algorithm is described in Ashton, M. [1]_,implementation was created by Roger Sayle [3]_ [4]_ [5]_
    at the 2017 RDKit user group meeting [2]_.[4]_

    It starts by vectorizing molecules with ECFP4 fingerprint. First test molecule is picked randomly. Each next one
    is selected to maximize the minimal distance to the already selected molecules (hence the MaxMin name) [4]_.,
    calculating and recording the distances as required. This molecule is the most distant
    one to those already picked so is transferred to the picked set [3]_.

    Parameters
    ----------
    data : sequence
        A sequence representing either SMILES strings or RDKit `Mol` objects.
    additional_data: list[sequence]
        Additional sequences to be split alongside the main data (e.g., labels or feature vectors).
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
    .. [1] `Mark Ashton, John Barnardb, Florence Casset, Michael Charlton, Geoffrey Downsb,
        Dominique Gorse, JohnHolliday, Roger Lahanac, Peter Willett
        "Identification of Diverse Database Subsets using Property-Based and Fragment-Based Molecular Descriptions"
        <https://onlinelibrary.wiley.com/doi/10.1002/qsar.200290002>_`
    .. [2] `Roger Sayle
        "Improved RDKit implementation"
        <https://github.com/rdkit/UGM_2017/blob/master/Presentations/Sayle_RDKitDiversity_Berlin17.pdf>_`
    .. [3] `Tim Dudgeon
        "Revisting the MaxMinPicker"
        <https://rdkit.org/docs/cppapi/classRDPickers_1_1MaxMinPicker.html>_`
    .. [4] `RDKit MaxMin Picker
        <https://squonk.it/docs/cells/RDKit%20MaxMin%20Picker>_`
    """

    data_size = len(data)
    train_size, valid_size, test_size = validate_train_valid_test_split_sizes(
        train_size, valid_size, test_size, len(data)
    )
    mols = ensure_mols(data)
    fps = GetMorganGenerator().GetFingerprints(mols)

    picker = MaxMinPicker()
    # Select only test set, starting from random pool
    test_idxs = picker.LazyBitVectorPick(
        fps,
        poolSize=data_size,
        pickSize=test_size,
        seed=random_state,
    )

    # firstPicks - initial state of picked set, so picked set will be union on test and valid sets
    # And then the previously selected test set is subtracted
    valid_idxs = picker.LazyBitVectorPick(
        fps,
        poolSize=data_size,
        pickSize=test_size + valid_size,
        firstPicks=test_idxs,
        seed=random_state,
    )
    # Rest elements are picked to train set
    train_idxs = list(set(range(data_size)) - set(test_idxs) - set(valid_idxs))
    # Substracting test set from joined sets
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
