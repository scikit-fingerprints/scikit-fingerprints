import warnings
from collections import defaultdict
from collections.abc import Sequence
from itertools import chain
from numbers import Integral
from typing import Any, Optional, Union

import numpy as np
from rdkit.Chem import Mol
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
from sklearn.utils._param_validation import Interval, RealNotInt, validate_params

from skfp.model_selection.utils import (
    ensure_nonempty_list,
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
    random_state:int = 0
) -> Union[
    tuple[
        Sequence[Union[str, Mol]], Sequence[Union[str, Mol]], Sequence[Sequence[Any]]
    ],
    tuple[Sequence, ...],
    tuple[Sequence[int], Sequence[int]],
]:
    data_size = len(data)
    train_size, test_size = validate_train_test_sizes(train_size, test_size)
    molecules = ensure_mols(data)
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024) for mol in molecules]
    picker = MaxMinPicker()
    test_idxs = set(picker.LazyPick(distFunc= lambda i,j : 1 - DataStructs.DiceSimilarity(fingerprints[i], fingerprints[j]),
                                      poolSize=data_size,
                                      pickSize=int(test_size * data_size),
                                      seed = random_state
                                      ))
    
    train_idxs = set(range(data_size)) - test_idxs
    train_subset: list[Any] = []
    test_subset: list[Any] = []

    if return_indices:
        train_subset = train_idxs
        test_subset = test_idxs
    else:
        train_subset = get_data_from_indices(data, train_idxs)
        test_subset = get_data_from_indices(data, test_idxs)

    ensure_nonempty_list(train_subset)
    ensure_nonempty_list(test_subset)

    additional_data_split: list[Sequence[Any]] = []

    additional_data_split = (
        split_additional_data(list(additional_data), [train_idxs, test_idxs])
        if additional_data
        else []
    )

    if additional_data:
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
    random_state:int = 0
) -> Union[
    tuple[
        Sequence[Union[str, Mol]], Sequence[Union[str, Mol]], Sequence[Sequence[Any]]
    ],
    tuple[Sequence, ...],
    tuple[Sequence[int], Sequence[int]],
]:
    data_size = len(data)
    train_size, valid_size, test_size = validate_train_valid_test_split_sizes(train_size, valid_size, test_size)
    molecules = ensure_mols(data)
    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024) for mol in molecules]
    picker = MaxMinPicker()
    test_idxs = set(picker.LazyPick(distFunc= lambda i,j : 1 - DataStructs.DiceSimilarity(fingerprints[i], fingerprints[j]),
                                      poolSize=data_size,
                                      pickSize=int(test_size * data_size),
                                      seed = random_state
                                      ))
    
    valid_idxs = test_idxs = set(picker.LazyPick(distFunc= lambda i,j : 1 - DataStructs.DiceSimilarity(fingerprints[i], fingerprints[j]),
                                      poolSize=data_size,
                                      pickSize=int((test_size + valid_size) * data_size),
                                      firstPicks=test_idxs,
                                      seed = random_state
                                      ))
    
    train_idxs = set(range(data_size)) - (test_idxs + valid_idxs)
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
    ensure_nonempty_list(train_subset)
    ensure_nonempty_list(test_subset)

    additional_data_split: list[Sequence[Any]] = []

    if len(valid_subset) == 0:
        warnings.warn(
            "Warning: Valid subset is empty. Consider using scaffold_train_test_split instead."
        )

    if additional_data:
        additional_data_split: list[Sequence[Any]] = split_additional_data(
            list(additional_data), train_idxs, valid_idxs, test_idxs
        )
        return train_subset, valid_subset, test_subset, additional_data_split
    else:
        return train_subset, valid_subset, test_subset
    
