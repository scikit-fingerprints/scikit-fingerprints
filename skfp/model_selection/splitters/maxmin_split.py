from collections.abc import Sequence
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd
from rdkit.Chem import Mol
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
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
        "return_indices": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def maxmin_train_test_split(
    data: Sequence[str | Mol],
    *additional_data: Sequence,
    train_size: float | None = None,
    test_size: float | None = None,
    return_indices: bool = False,
    random_state: int = 0,
) -> (
    tuple[Sequence[str | Mol], Sequence[str | Mol], Sequence[Sequence[Any]]]
    | tuple[Sequence, ...]
    | tuple[Sequence[int], Sequence[int]]
):
    """
    Split using MaxMin algorithm.

    MaxMinPicker is an efficient algorithm for picking an optimal subset of diverse
    compounds from a candidate pool. The original algorithm was introduced in [1]_,
    but here we use an optimized implementation by Roger Sayle [2]_ [3]_ [4]_.

    First, molecules are vectorized using binary ECFP4 fingerprint (radius 2) with
    2048 bits. The first test molecule is picked randomly. Each next one is selected
    to maximize the minimal distance to the already selected molecules (hence the
    MaxMin name) [4]_. Distances are calculated on the fly as required.

    First, the test set is constructed, and training set are all other molecules.

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

    return_indices : bool, default=False
        Whether the method should return the input object subsets, i.e. SMILES strings
        or RDKit ``Mol`` objects, or only the indices of the subsets instead of the data.

    random_state: int, default=0
        Random generator seed that will be used for selecting initial molecules.

    Returns
    -------
    subsets : tuple[list, list, ...]
        Tuple with train-test subsets of provided arrays. First two are lists of SMILES
        strings or RDKit ``Mol`` objects, depending on the input type. If `return_indices`
        is True, lists of indices are returned instead of actual data.

    References
    ----------
    .. [1] `Mark Ashton et al.
        "Identification of Diverse Database Subsets using Property-Based and Fragment-Based Molecular Descriptions"
        Quant. Struct.-Act. Relat., 21: 598-604
        <https://onlinelibrary.wiley.com/doi/10.1002/qsar.200290002>_`

    .. [2] `Roger Sayle
        "Improved RDKit implementation"
        <https://github.com/rdkit/UGM_2017/blob/master/Presentations/Sayle_RDKitDiversity_Berlin17.pdf>_`

    .. [3] `Tim Dudgeon
        "Revisiting the MaxMinPicker"
        <https://rdkit.org/docs/cppapi/classRDPickers_1_1MaxMinPicker.html>_`

    .. [4] `Squonk - RDKit MaxMin Picker
        <https://squonk.it/docs/cells/RDKit%20MaxMin%20Picker>_`
    """
    data_size = len(data)
    train_size, test_size = validate_train_test_split_sizes(
        train_size, test_size, data_size
    )

    mols = ensure_mols(data)
    fps = GetMorganGenerator(radius=2, fpSize=2048).GetFingerprints(mols)

    picker = MaxMinPicker()
    test_idxs = picker.LazyBitVectorPick(
        fps,
        poolSize=data_size,
        pickSize=test_size,
        seed=random_state,
    )
    test_idxs = list(test_idxs)
    train_idxs = list(set(range(data_size)) - set(test_idxs))

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
        "return_indices": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def maxmin_train_valid_test_split(
    data: Sequence[str | Mol],
    *additional_data: Sequence,
    train_size: float | None = None,
    valid_size: float | None = None,
    test_size: float | None = None,
    return_indices: bool = False,
    random_state: int = 0,
) -> (
    tuple[Sequence[str | Mol], Sequence[str | Mol], Sequence[Sequence[Any]]]
    | tuple[Sequence, ...]
    | tuple[Sequence[int], Sequence[int]]
):
    """
    Split using MaxMin algorithm.

    MaxMinPicker is an efficient algorithm for picking an optimal subset of diverse
    compounds from a candidate pool. The original algorithm was introduced in [1]_,
    but here we use an optimized implementation by Roger Sayle [2]_ [3]_ [4]_.

    First, molecules are vectorized using binary ECFP4 fingerprint (radius 2) with
    2048 bits. The first test molecule is picked randomly. Each next one is selected
    to maximize the minimal distance to the already selected molecules (hence the
    MaxMin name) [4]_. Distances are calculated on the fly as required.

    First, the test set is constructed, then validation, and training set are all
    other molecules.

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

    return_indices : bool, default=False
        Whether the method should return the input object subsets, i.e. SMILES strings
        or RDKit ``Mol`` objects, or only the indices of the subsets instead of the data.

    random_state: int, default=0
        Random generator seed that will be used for selecting initial molecules.

    Returns
    -------
    subsets : tuple[list, list, ...]
        Tuple with train-valid-test subsets of provided arrays. First three are lists of
        SMILES strings or RDKit ``Mol`` objects, depending on the input type. If
        `return_indices` is True, lists of indices are returned instead of actual data.

    References
    ----------
    .. [1] `Mark Ashton et al.
        "Identification of Diverse Database Subsets using Property-Based and Fragment-Based Molecular Descriptions"
        Quant. Struct.-Act. Relat., 21: 598-604
        <https://onlinelibrary.wiley.com/doi/10.1002/qsar.200290002>_`

    .. [2] `Roger Sayle
        "Improved RDKit implementation"
        <https://github.com/rdkit/UGM_2017/blob/master/Presentations/Sayle_RDKitDiversity_Berlin17.pdf>_`

    .. [3] `Tim Dudgeon
        "Revisiting the MaxMinPicker"
        <https://rdkit.org/docs/cppapi/classRDPickers_1_1MaxMinPicker.html>_`

    .. [4] `Squonk - RDKit MaxMin Picker
        <https://squonk.it/docs/cells/RDKit%20MaxMin%20Picker>_`
    """
    data_size = len(data)
    train_size, valid_size, test_size = validate_train_valid_test_split_sizes(
        train_size, valid_size, test_size, len(data)
    )
    mols = ensure_mols(data)
    fps = GetMorganGenerator(radius=2, fpSize=2048).GetFingerprints(mols)

    picker = MaxMinPicker()

    # select the test set only
    test_idxs = picker.LazyBitVectorPick(
        fps,
        poolSize=data_size,
        pickSize=test_size,
        seed=random_state,
    )
    test_idxs = list(test_idxs)

    # select validation + test sets, first including already computed test set
    # then remove test indexes, leaving only validation set
    valid_idxs = picker.LazyBitVectorPick(
        fps,
        poolSize=data_size,
        pickSize=test_size + valid_size,
        firstPicks=test_idxs,
        seed=random_state,
    )
    valid_idxs = list(set(valid_idxs) - set(test_idxs))

    train_idxs = list(set(range(data_size)) - set(test_idxs) - set(valid_idxs))

    ensure_nonempty_subset(train_idxs, "train")
    ensure_nonempty_subset(valid_idxs, "valid")
    ensure_nonempty_subset(test_idxs, "test")

    if return_indices:
        train_subset = train_idxs
        test_subset = test_idxs
        valid_subset = valid_idxs
    else:
        train_subset = get_data_from_indices(data, train_idxs)
        test_subset = get_data_from_indices(data, test_idxs)
        valid_subset = get_data_from_indices(data, valid_idxs)

    if additional_data:
        additional_data_split: list[Sequence[Any]] = split_additional_data(
            list(additional_data), train_idxs, valid_idxs, test_idxs
        )
        return train_subset, valid_subset, test_subset, *additional_data_split
    else:
        return train_subset, valid_subset, test_subset


@validate_params(
    {
        "data": ["array-like"],
        "labels": ["array-like"],
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
def maxmin_stratified_train_test_split(
    data: Sequence[str | Mol],
    labels: np.ndarray | list[int] | pd.Series,
    *additional_data: Sequence,
    train_size: float | None = None,
    test_size: float | None = None,
    return_indices: bool = False,
    random_state: int = 0,
) -> (
    tuple[Sequence[str | Mol], Sequence[str | Mol], Sequence[Sequence[Any]]]
    | tuple[Sequence, ...]
    | tuple[Sequence[int], Sequence[int]]
):
    """
    Split using MaxMin algorithm with stratification.

    A variant of MaxMin split (see :py:func:`maxmin_train_test_split`), modified to
    split each class separately. The goal is to preserve the class distribution in
    the resulting train and test splits, while also distributing points in each subset
    across the chemical space.

    Note that results may differ quite strongly from regular MaxMin split, as here
    classes are treated independently of each other. While the distances between
    compounds in each class are maximized, there are no guarantees for the overall
    dataset. However, generally this split should also result in relatively uniform
    coverage of the entire chemical space.

    Resulting sizes of train and test sets may differ slightly for very small datasets.
    This is because each class is split separately with a given percentage.

    If ``train_size`` and ``test_size`` are integers, they must sum up to the ``data``
    length. If they are floating numbers, they must sum up to 1.

    Parameters
    ----------
    data : sequence
        A sequence representing either SMILES strings or RDKit ``Mol`` objects.

    labels : array-like
        An array or list with class labels as integers.

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
        or RDKit ``Mol`` objects, or only the indices of the subsets instead of the data.

    random_state: int, default=0
        Random generator seed that will be used for selecting initial molecules.

    Returns
    -------
    subsets : tuple[list, list, ...]
        Tuple with train-test subsets of provided arrays. First two are lists of SMILES
        strings or RDKit ``Mol`` objects, depending on the input type. Third and fourth
        are NumPy arrays with labels of train and test subsets. If `return_indices` is
        True, lists of indices are returned instead of actual data as the first two
        elements.

    See Also
    --------
    :func:`maxmin_train_test_split` : Regular MaxMin split.
    """
    data_arr = np.array(data)
    labels = np.array(labels, dtype=int)

    train_idxs = []
    test_idxs = []

    for label in np.unique(labels):
        label_idxs = np.nonzero(labels == label)[0]
        label_data = data_arr[label_idxs]

        # split indices of current label into train and test
        # then map them to indices of the entire dataset
        label_train_idxs, label_test_idxs = maxmin_train_test_split(
            label_data,
            train_size=train_size,
            test_size=test_size,
            return_indices=True,
            random_state=random_state,
        )
        label_train_idxs = label_idxs[label_train_idxs].tolist()
        label_test_idxs = label_idxs[label_test_idxs].tolist()

        train_idxs.extend(label_train_idxs)
        test_idxs.extend(label_test_idxs)

    if return_indices:
        train_subset = train_idxs
        test_subset = test_idxs
    else:
        train_subset = get_data_from_indices(data, train_idxs)
        test_subset = get_data_from_indices(data, test_idxs)

    labels_train = labels[train_idxs]
    labels_test = labels[test_idxs]

    if additional_data:
        additional_data_split: list[Sequence[Any]] = split_additional_data(
            list(additional_data), train_idxs, test_idxs
        )
        return (
            train_subset,
            test_subset,
            labels_train,
            labels_test,
            *additional_data_split,
        )
    else:
        return train_subset, test_subset, labels_train, labels_test


@validate_params(
    {
        "data": ["array-like"],
        "labels": ["array-like"],
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
        "return_indices": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def maxmin_stratified_train_valid_test_split(
    data: Sequence[str | Mol],
    labels: np.ndarray | list[int] | pd.Series,
    *additional_data: Sequence,
    train_size: float | None = None,
    valid_size: float | None = None,
    test_size: float | None = None,
    return_indices: bool = False,
    random_state: int = 0,
) -> (
    tuple[Sequence[str | Mol], Sequence[str | Mol], Sequence[Sequence[Any]]]
    | tuple[Sequence, ...]
    | tuple[Sequence[int], Sequence[int]]
):
    """
    Split using MaxMin algorithm with stratification.

    A variant of MaxMin split (see :py:func:`maxmin_train_valid_test_split`), modified
    to split each class separately. The goal is to preserve the class distribution in
    the resulting train, valid, and test splits, while also distributing points in each
    subset across the chemical space.

    Note that results may differ quite strongly from regular MaxMin split, as here
    classes are treated independently of each other. While the distances between
    compounds in each class are maximized, there are no guarantees for the overall
    dataset. However, generally this split should also result in relatively uniform
    coverage of the entire chemical space.

    Resulting sizes of train, valid, and test sets may differ slightly for very small
    datasets. This is because each class is split separately with a given percentage.

    If ``train_size``, ``valid_size`` and ``test_size`` are integers, they must sum up
    to the ``data`` length. If they are floating numbers, they must sum up to 1.

    Parameters
    ----------
    data : sequence
        A sequence representing either SMILES strings or RDKit ``Mol`` objects.

    labels : array-like
        An array or list with class labels as integers.

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

    return_indices : bool, default=False
        Whether the method should return the input object subsets, i.e. SMILES strings
        or RDKit ``Mol`` objects, or only the indices of the subsets instead of the data.

    random_state: int, default=0
        Random generator seed that will be used for selecting initial molecules.

    Returns
    -------
    subsets : tuple[list, list, ...]
        Tuple with train-valid-test subsets of provided arrays. First three are lists of
        SMILES strings or RDKit ``Mol`` objects, depending on the input type. Next three
        are NumPy arrays with labels of train-valid-test subsets. If `return_indices` is
        True, lists of indices are returned instead of actual data as the first three
        elements.

    See Also
    --------
    :func:`maxmin_train_valid_test_split` : Regular MaxMin split.
    """
    data_arr = np.array(data)
    labels = np.array(labels, dtype=int)

    train_idxs = []
    valid_idxs = []
    test_idxs = []

    for label in np.unique(labels):
        label_idxs = np.nonzero(labels == label)[0]
        label_data = data_arr[label_idxs]

        # split indices of current label into train and test
        # then map them to indices of the entire dataset
        label_train_idxs, label_valid_idxs, label_test_idxs = (
            maxmin_train_valid_test_split(
                label_data,
                train_size=train_size,
                valid_size=valid_size,
                test_size=test_size,
                return_indices=True,
                random_state=random_state,
            )
        )
        label_train_idxs = label_idxs[label_train_idxs].tolist()
        label_valid_idxs = label_idxs[label_valid_idxs].tolist()
        label_test_idxs = label_idxs[label_test_idxs].tolist()

        train_idxs.extend(label_train_idxs)
        valid_idxs.extend(label_valid_idxs)
        test_idxs.extend(label_test_idxs)

    if return_indices:
        train_subset = train_idxs
        valid_subset = valid_idxs
        test_subset = test_idxs
    else:
        train_subset = get_data_from_indices(data, train_idxs)
        valid_subset = get_data_from_indices(data, valid_idxs)
        test_subset = get_data_from_indices(data, test_idxs)

    labels_train = labels[train_idxs]
    labels_valid = labels[valid_idxs]
    labels_test = labels[test_idxs]

    if additional_data:
        additional_data_split: list[Sequence[Any]] = split_additional_data(
            list(additional_data), train_idxs, valid_idxs, test_idxs
        )
        return (
            train_subset,
            valid_subset,
            test_subset,
            labels_train,
            labels_valid,
            labels_test,
            *additional_data_split,
        )
    else:
        return (
            train_subset,
            valid_subset,
            test_subset,
            labels_train,
            labels_valid,
            labels_test,
        )
