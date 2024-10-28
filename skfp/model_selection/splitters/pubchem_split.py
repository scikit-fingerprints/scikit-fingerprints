import json
import time
import warnings
from collections.abc import Sequence
from numbers import Integral
from typing import Any, Optional, Union
from urllib.parse import quote

import pandas as pd
import requests
from rdkit.Chem import Mol, MolToSmiles
from sklearn.utils._param_validation import (
    Interval,
    RealNotInt,
    StrOptions,
    validate_params,
)

from skfp.model_selection.splitters.utils import (
    ensure_nonempty_subset,
    get_data_from_indices,
    run_in_parallel,
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
        "not_found_behavior": [StrOptions({"train", "test", "remove"})],
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
    n_jobs: int = 5,
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

    Note that this split can be very slow for large number of molecules, since requests
    to PubChem frequently fail due to the "servers overloaded" error.

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

    n_jobs : int, default=5
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
    .. [1] `Kim S, Thiessen PA, Cheng T, Yu B, Bolton EE.
        "An update on PUG-REST: RESTful interface for programmatic access to PubChem."
        Nucleic Acids Res. 2018 Jul 2;46(W1):W563-W570.
        <https://doi.org/10.1093/nar/gky294>`_
    """
    years = _get_pubchem_years(data, n_jobs)
    data_df = pd.DataFrame({"idx": list(range(len(data))), "year": years})

    if not_found_behavior != "remove":
        not_found_mols = list(data_df[data_df["year"].isna()].idx)

    data_df = data_df[~data_df["year"].isna()]

    data_df = data_df.sort_values("year")

    train_size, test_size = validate_train_test_split_sizes(
        train_size, test_size, len(data_df)
    )

    train_idxs: list[int] = list(data_df.iloc[:train_size].idx)
    test_idxs: list[int] = list(data_df.iloc[train_size:].idx)

    if not_found_behavior == "train":
        train_idxs.extend(not_found_mols)

    if not_found_behavior == "test":
        test_idxs.extend(not_found_mols)

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
        "not_found_behavior": [StrOptions({"train", "test", "remove"})],
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
    n_jobs: int = 5,
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

    Note that this split can be very slow for large number of molecules, since requests
    to PubChem frequently fail due to the "servers overloaded" error.

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

    n_jobs : int, default=5
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
    .. [1] `Kim S, Thiessen PA, Cheng T, Yu B, Bolton EE.
        "An update on PUG-REST: RESTful interface for programmatic access to PubChem."
        Nucleic Acids Res. 2018 Jul 2;46(W1):W563-W570.
        <https://doi.org/10.1093/nar/gky294>`_
    """
    years = _get_pubchem_years(data, n_jobs)

    data_df = pd.DataFrame({"idx": list(range(len(data))), "year": years})

    if not_found_behavior != "remove":
        not_found_mols = list(data_df[data_df["year"].isna()].idx)

    data_df = data_df[~data_df["year"].isna()]
    data_df = data_df.sort_values("year")

    train_size, valid_size, test_size = validate_train_valid_test_split_sizes(
        train_size, valid_size, test_size, len(data)
    )

    train_idxs: list[int] = list(data_df.iloc[:train_size].idx)
    valid_idxs: list[int] = list(data_df.iloc[train_size : train_size + valid_size].idx)
    test_idxs: list[int] = list(data_df.iloc[train_size + valid_size :].idx)

    if not_found_behavior == "train":
        train_idxs.extend(not_found_mols)

    if not_found_behavior == "test":
        test_idxs.extend(not_found_mols)

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


def _get_pubchem_years(
    data: Sequence[Union[str, Mol]], n_jobs: int = 5
) -> list[list[int]]:
    """
    Get first literature publication year from PubChem for a list of molecules, either
    as SMILES strings or RDKit Mol objects.

    Parameters
    ----------
    data : sequence
        A sequence representing either SMILES strings or RDKit `Mol` objects.
    """
    data = list(
        map(
            lambda mol: (
                MolToSmiles(mol, canonical=True, isomericSmiles=True)
                if mol is Mol
                else mol
            ),
            data,
        )
    )

    print("Converting SMILES to CIDs")
    cids = run_in_parallel(get_cid_for_smiles, data, n_jobs)

    print("Searching for CIDs literature")
    years = run_in_parallel(get_earliest_publication_date, cids, n_jobs)

    return years


def get_cid_for_smiles(smiles: str) -> Optional[str]:
    """

    Parameters:
    ------------
    smiles: str
        SMILES string

    Return
    --------
        PubChem compound identifier if  else None

    """
    PUG_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{quote(smiles)}/cids/JSON"

    PUG_response = requests.get(PUG_url, timeout=10)
    PUG_response_status = PUG_response.status_code

    if PUG_response_status != 200:
        warnings.warn(
            f"PUG error \n, {PUG_url} \n status code {PUG_response_status}, \n check your input"
        )

    PUG_response_json = PUG_response.json()

    if "IdentifierList" in PUG_response_json:
        cid = PUG_response_json["IdentifierList"]["CID"][0]
        if bool(cid):
            cid = str(cid)
        else:
            cid = None
    else:
        cid = None
    return cid


def get_earliest_publication_date(cid: Union[str, None]) -> Optional[int]:
    """
    Get the date of the earliest publication from PubChem where a given molecule appears.
    There are molecules without publications, for which we return None.

    Parameters
    ----------
    Optional[int]
        year of earliest publication date associated to cid

    Returns
    ----------

    """
    if not cid:
        return None

    query = {
        "download": "*",
        "collection": "literature",
        "where": {"ands": [{"cid": str(cid)}]},
        "order": ["articlepubdate,asc"],
        "start": 0,
        "limit": 3,  # fetch a few, just in case
        "downloadfilename": f"pubchem_cid_{cid}_literature",
        "nullatbottom": 1,
    }
    base_url = "https://pubchem.ncbi.nlm.nih.gov/sdq/sdqagent.cgi"
    params = {"infmt": "json", "outfmt": "json", "query": json.dumps(query)}

    def get_publication_date(response: list[dict]) -> Optional[int]:
        article_pub_date = None
        for entry in response:
            if "articlepubdate" in entry:
                return int(entry["articlepubdate"][:4])  # get year of publication
            else:
                continue
        return article_pub_date

    while True:
        try:
            response = requests.get(base_url, params=params, timeout=10).json()
            if isinstance(response, list):
                if len(response) >= 1:
                    date = get_publication_date(response)
                    return date
                else:
                    # no literature found
                    return None
            else:
                # most probably "Server too busy" error
                time.sleep(1)
        except requests.ConnectionError:
            # most probably unstable PubChem network
            time.sleep(1)
