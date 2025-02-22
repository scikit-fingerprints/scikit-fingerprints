import json
import sys
import time
import warnings
from collections.abc import Sequence
from numbers import Integral
from typing import Any, Optional, Union
from urllib.parse import quote

import requests
from joblib.parallel import effective_n_jobs
from rdkit.Chem import Mol
from sklearn.utils._param_validation import (
    Interval,
    RealNotInt,
    StrOptions,
    validate_params,
)

from skfp.model_selection.splitters.utils import (
    ensure_nonempty_subset,
    split_additional_data,
    validate_train_test_split_sizes,
    validate_train_valid_test_split_sizes,
)
from skfp.utils.functions import get_data_from_indices
from skfp.utils.parallel import run_in_parallel
from skfp.utils.validators import ensure_smiles


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
    n_retries: int = 3,
    verbose: int = 0,
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

    n_retries : int, default=3
        defines the number of re-requests to the PubChem REST API in case of errors
        during a request.

    n_jobs : int, default=5
        The number of jobs to run in parallel for fetching data from PubChem. Note that
        since those are requests relying on I/O, threads are used instead of processes.
        At most 5 requests per second are made to avoid throttling, which limits
        parallelism even if high ``n_jobs`` is set.

    verbose : int, default=0
        Controls the verbosity when fetching data from PubChem.

    Returns
    -------
    subsets : tuple[list, list, ...]
        Tuple with train-test subsets of provided arrays. First two are lists of SMILES
        strings or RDKit ``Mol`` objects, depending on the input type. If ``return_indices``
        is True, lists of indices are returned instead of actual data.

    References
    ----------
    .. [1] `Kim S, Thiessen PA, Cheng T, Yu B, Bolton EE.
        "An update on PUG-REST: RESTful interface for programmatic access to PubChem."
        Nucleic Acids Res. 2018 Jul 2;46(W1):W563-W570.
        <https://doi.org/10.1093/nar/gky294>`_
    """
    years = _get_pubchem_years(data, n_jobs, n_retries, verbose)

    idxs_with_year = [(year, i) for i, year in enumerate(years) if year is not None]
    idxs_with_year.sort()
    idxs_with_year = [i for year, i in idxs_with_year]

    idxs_no_year = [i for i, year in enumerate(years) if year is None]

    train_size, test_size = validate_train_test_split_sizes(
        train_size, test_size, len(idxs_with_year)
    )

    train_idxs = idxs_with_year[:train_size]
    test_idxs = idxs_with_year[train_size:]

    if not_found_behavior == "train":
        train_idxs.extend(idxs_no_year)
    elif not_found_behavior == "test":
        test_idxs.extend(idxs_no_year)

    ensure_nonempty_subset(train_idxs, "train")
    ensure_nonempty_subset(test_idxs, "test")

    if return_indices:
        train_subset = train_idxs
        test_subset = test_idxs
    else:
        train_subset = get_data_from_indices(data, train_idxs)
        test_subset = get_data_from_indices(data, test_idxs)

    if additional_data:
        additional_data_split = split_additional_data(
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
    n_retries: int = 3,
    n_jobs: int = 5,
    verbose: int = 1,
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

    n_retries : int, default=3
        Define number of request retries to the PubChem REST API in case of errors
        during a request.

    n_jobs : int, default=5
        The number of jobs to run in parallel for fetching data from PubChem. Note that
        since those are requests relying on I/O, threads are used instead of processes.
        At most 5 requests per second are made to avoid throttling, which limits
        parallelism even if high ``n_jobs`` is set.

    verbose : int, default=0
        Controls the verbosity when fetching data from PubChem.

    Returns
    -------
    subsets : tuple[list, list, ...]
        Tuple with train-valid-test subsets of provided arrays. First three are lists of
        SMILES strings or RDKit ``Mol`` objects, depending on the input type. If ``return_indices``
        is True, lists of indices are returned instead of actual data.

    References
    ----------
    .. [1] `Kim S, Thiessen PA, Cheng T, Yu B, Bolton EE.
        "An update on PUG-REST: RESTful interface for programmatic access to PubChem."
        Nucleic Acids Res. 2018 Jul 2;46(W1):W563-W570.
        <https://doi.org/10.1093/nar/gky294>`_
    """
    years = _get_pubchem_years(data, n_jobs, n_retries, verbose)

    idxs_with_year = [(year, i) for i, year in enumerate(years) if year is not None]
    idxs_with_year.sort()
    idxs_with_year = [i for year, i in idxs_with_year]

    idxs_no_year = [i for i, year in enumerate(years) if year is None]

    train_size, valid_size, test_size = validate_train_valid_test_split_sizes(
        train_size, valid_size, test_size, len(data)
    )

    train_idxs = idxs_with_year[:train_size]
    valid_idxs = idxs_with_year[train_size : train_size + valid_size]
    test_idxs = idxs_with_year[train_size + valid_size :]

    if not_found_behavior == "train":
        train_idxs.extend(idxs_no_year)
    elif not_found_behavior == "test":
        test_idxs.extend(idxs_no_year)

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
        additional_data_split = split_additional_data(
            list(additional_data), train_idxs, valid_idxs, test_idxs
        )
        return train_subset, valid_subset, test_subset, *additional_data_split
    else:
        return train_subset, valid_subset, test_subset


def _get_pubchem_years(
    data: Sequence[Union[str, Mol]], n_jobs: int, n_retries: int, verbosity: int
) -> list[Optional[int]]:
    """
    Get first literature publication year from PubChem for a list of molecules, either
    as SMILES strings or RDKit Mol objects.
    """
    if n_retries == -1:
        n_retries = sys.maxsize
    n_jobs = min(effective_n_jobs(n_jobs), 5)

    data = ensure_smiles(data)

    if verbosity > 0:
        print("Converting SMILES to CIDs")
    cids = run_in_parallel(
        _get_cid_for_smiles,
        data,
        n_jobs,
        verbosity=verbosity,
        n_retries=n_retries,
        single_element_func=True,
    )

    if verbosity > 0:
        print("Searching for literature data")
    years = run_in_parallel(
        _get_earliest_publication_date,
        cids,
        n_jobs,
        n_retries=n_retries,
        single_element_func=True,
    )

    return years


def _get_cid_for_smiles(smiles: str, n_retries: int, verbosity: int) -> Optional[str]:
    """
    Get PubChem CID from SMILES, or None if molecule cannot be found.
    """
    print(smiles)
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{quote(smiles)}/cids/JSON"

    response = None
    trial = 0
    while trial < n_retries:
        try:
            response = requests.get(url, timeout=20)
            response_status = response.status_code
            if response_status != 200 and verbosity > 0:
                warnings.warn(
                    f"PUG error for URL {url}, status code {response_status}, description: {response.text}"
                )

            if response:
                response_json = response.json()
                cid = response_json["IdentifierList"]["CID"][0]
                cid = str(cid) if cid else None
                return cid
            else:
                trial += 1

        except (requests.exceptions.Timeout, requests.exceptions.RequestException):
            time.sleep(1)
            trial += 1
        except (KeyError, ValueError):
            return None
    raise RuntimeError("CID could not be downloaded in {n_retries} retries")


def _get_earliest_publication_date(cid: Optional[str], n_retries: int) -> Optional[int]:
    """
    Get the date of the earliest publication from PubChem where a given molecule appears.
    There are molecules without publications, for which we return None.
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
        return article_pub_date

    trial = 0
    while trial < n_retries:
        try:
            response = requests.get(base_url, params=params, timeout=10)
            status_code = response.status_code
            response = response.json()

            if status_code != 200:
                trial += 1
                continue

            if isinstance(response, list) and len(response) >= 1:
                date = get_publication_date(response)
                return date
            else:
                # most probably "Server too busy" error
                time.sleep(1)

        except requests.ConnectionError:
            # most probably unstable PubChem network
            time.sleep(1)
        except requests.exceptions.Timeout:
            time.sleep(1)
        trial += 1

    return None
