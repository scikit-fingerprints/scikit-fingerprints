from collections.abc import Sequence
from numbers import Integral
from typing import Any, Optional, Union

from rdkit.Chem import Mol
from sklearn.utils._param_validation import Interval, RealNotInt, validate_params

from skfp.model_selection.splitters.utils import (
    ensure_nonempty_subset,
    get_data_from_indices,
    split_additional_data,
    validate_train_test_split_sizes,
    validate_train_valid_test_split_sizes,
)


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
        "not_found_behavior": ["train", "test", "remove"],
        "return_type": ["same_as_input", "indices", "dataframe"],
        "n_jobs": [Integral, None],
    },
    prefer_skip_nested_validation=True,
)
def pubchem_train_test_split(
    data: Sequence[Union[str, Mol]],
    *additional_data: Sequence,
    train_size: Optional[float] = None,
    test_size: Optional[float] = None,
    not_found_behavior: str = "test",
    return_indices: bool = False,
    n_jobs: Optional[int] = None,
) -> Union[
    tuple[
        Sequence[Union[str, Mol]], Sequence[Union[str, Mol]], Sequence[Sequence[Any]]
    ],
    tuple[Sequence, ...],
    tuple[Sequence[int], Sequence[int]],
]:
    """
    Split using PubChem literature data.

    This is a time (chronological) split, using first literature date in PubChem
    available for each molecule. Molecules are partitioned deterministically, with
    the newest ones assigned to the test subset and the rest to the training subset.

    In case of no data available (no literature or no PubChem entry) the behavior
    is governed by the ``not_found_behavior`` parameter. Full DataFrame with columns
    ``["SMILES", "split", "year"]`` can also be returned.

    The split fractions (train_size, test_size) must sum to 1.

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

    not_found_behavior : {"train", "test", "remove"}, default="test"
        What to do with molecules not found in PubChem, or if molecule does not have
        any literature data available. Such molecules can be put into train set,
        test set, or removed altogether.

    return_indices : bool, default=False
        Whether the method should return the input object subsets, i.e. SMILES strings
        or RDKit ``Mol`` objects, or only the indices of the subsets instead of the data.

    n_jobs : int, default=None
        The number of jobs to run in parallel for fetching data from PubChem. Note that
        since those are requests relying on I/O, threads are used instead of processes.
        At most 5 requests per second are made to avoid throttling, which limits
        parallelism even if high ``n_jobs`` is set.

    Returns
    ----------
    subsets : tuple[list, list, ...]
    Tuple with train-test subsets of provided arrays. First two are lists of SMILES
    strings or RDKit `Mol` objects, depending on the input type. If `return_indices`
    is True, lists of indices are returned instead of actual data.

    References
    ----------
    .. [1]
    """
    train_size, test_size = validate_train_test_split_sizes(
        train_size, test_size, len(data)
    )

    years = _get_pubchem_years(data, n_jobs)
    ...  # probably use Pandas or lists here

    train_idxs: list[int] = []
    test_idxs: list[int] = []

    # time split, include not_found_behavior parameter
    ...

    ensure_nonempty_subset(train_idxs, "train")
    ensure_nonempty_subset(test_idxs, "test")

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
        "not_found_behavior": ["train", "test", "remove"],
        "return_type": ["same_as_input", "indices", "dataframe"],
        "n_jobs": [Integral, None],
    },
    prefer_skip_nested_validation=True,
)
def pubchem_train_valid_test_split(
    data: Sequence[Union[str, Mol]],
    *additional_data: Sequence,
    train_size: Optional[float] = None,
    valid_size: Optional[float] = None,
    test_size: Optional[float] = None,
    not_found_behavior: str = "test",
    return_indices: bool = False,
    n_jobs: Optional[int] = None,
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
    Split using PubChem literature data.

    This is a time (chronological) split, using first literature date in PubChem
    available for each molecule. Molecules are partitioned deterministically, with
    the newest ones assigned to the test subset and the rest to the training subset.

    In case of no data available (no literature or no PubChem entry) the behavior
    is governed by the ``not_found_behavior`` parameter. Full DataFrame with columns
    ``["SMILES", "split", "year"]`` can also be returned.

    The split fractions (train_size, valid_size, test_size) must sum to 1.

    Parameters
    ----------
    data : sequence
        A sequence representing either SMILES strings or RDKit `Mol` objects.

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

    not_found_behavior : {"train", "test", "remove"}, default="test"
        What to do with molecules not found in PubChem, or if molecule does not have
        any literature data available. Such molecules can be put into train set,
        test set, or removed altogether.

    return_indices : bool, default=False
        Whether the method should return the input object subsets, i.e. SMILES strings
        or RDKit ``Mol`` objects, or only the indices of the subsets instead of the data.

    return_indices : bool, default=False
        Whether the method should return the input object subsets, i.e. SMILES strings
        or RDKit `Mol` objects, or only the indices of the subsets instead of the data.

    n_jobs : int, default=None
        The number of jobs to run in parallel for fetching data from PubChem. Note that
        since those are requests relying on I/O, threads are used instead of processes.
        At most 5 requests per second are made to avoid throttling, which limits
        parallelism even if high ``n_jobs`` is set.

    Returns
    ----------
    subsets : tuple[list, list, ...]
    Tuple with train-valid-test subsets of provided arrays. First three are lists of
    SMILES strings or RDKit `Mol` objects, depending on the input type. If `return_indices`
    is True, lists of indices are returned instead of actual data.

    References
    ----------
    .. [1]
    """
    train_size, valid_size, test_size = validate_train_valid_test_split_sizes(
        train_size, valid_size, test_size, len(data)
    )

    years = _get_pubchem_years(data, n_jobs)
    ...  # probably use Pandas or lists here

    train_idxs: list[int] = []
    valid_idxs: list[int] = []
    test_idxs: list[int] = []

    # time split, include not_found_behavior parameter
    ...

    if return_indices:
        train_subset = train_idxs
        valid_subset = valid_idxs
        test_subset = test_idxs
    else:
        train_subset = get_data_from_indices(data, train_idxs)
        valid_subset = get_data_from_indices(data, valid_idxs)
        test_subset = get_data_from_indices(data, test_idxs)

    ensure_nonempty_subset(train_subset, "train")
    ensure_nonempty_subset(valid_subset, "validation")
    ensure_nonempty_subset(test_subset, "test")

    if additional_data:
        additional_data_split: list[Sequence[Any]] = split_additional_data(
            list(additional_data), train_idxs, valid_idxs, test_idxs
        )
        return train_subset, valid_subset, test_subset, *additional_data_split
    else:
        return train_subset, valid_subset, test_subset


def _get_pubchem_years(data: Sequence[Union[str, Mol]], n_jobs: Optional[int] = None) -> list[list[int]]:
    """
    Get first literature publication year from PubChem for a list of molecules, either
    as SMILES strings or RDKit Mol objects.
    """

