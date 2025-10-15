import os
from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.utils._param_validation import StrOptions, validate_params

from .moleculeace import (
    load_chembl204_ki,
)

MOLECULEACE_DATASET_NAMES = [
    "chembl204_ki",
]

MOLECULEACE_DATASET_NAME_TO_LOADER_FUNC = {
    "chembl204_ki": load_chembl204_ki,
}


@validate_params(
    {
        "data_dir": [None, str, os.PathLike],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_moleculeace_benchmark(
    subset: str | list[str] | None = None,
    data_dir: str | os.PathLike | None = None,
    as_frames: bool = False,
    verbose: bool = False,
) -> Iterator[tuple[str, pd.DataFrame]] | Iterator[tuple[str, list[str], np.ndarray]]:
    """
    Load the MoleculeACE benchmark datasets.

    MoleculeACE [1]_ datasets are varied inhibition and effective concentration targets from ChEMBL [2]_.
    Activity cliff is recommended for all of them.

    For more details, see loading functions for particular datasets. Allowed individual
    dataset names are listed below. Dataset names are also returned (case-sensitive).

    - "chembl204_ki"

    Parameters
    ----------
    subset : None or list of strings
        If ``None``, returns all datasets. List of strings loads only datasets with given names.

    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If ``None``, currently set scikit-learn directory
        is used, by default `$HOME/scikit_learn_data`.

    as_frames : bool, default=False
        If True, returns the raw DataFrame for each dataset. Otherwise, returns SMILES
        as a list of strings, and labels as a NumPy array for each dataset.

    verbose : bool, default=False
        If True, progress bar will be shown for downloading or loading files.

    Returns
    -------
    data : generator of pd.DataFrame or tuples (list[str], np.ndarray)
        Loads and returns datasets with a generator. Returned types depend on the
        ``as_frame`` parameter, either:
        - Pandas DataFrame with columns: "SMILES", "label"
        - tuple of: list of strings (SMILES), NumPy array (labels)

    References
    ----------
    .. [1] `D. van Tilborg, A. Alenicheva, and F. Grisoni
        “Exposing the Limitations of Molecular Machine Learning with Activity Cliffs”
        Journal of Chemical Information and Modeling, vol. 62, no. 23, pp. 5938–5951, Dec. 2022.
        <https://doi.org/10.1021/acs.jcim.2c01073>`_
    .. [2] `B. Zdrazil et al.
        “The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods,”
        Nucleic Acids Research, vol. 52, no. D1, Nov. 2023.
        <https://doi.org/10.1093/nar/gkad1004>`_
    """
    dataset_names = _subset_to_dataset_names(subset)

    dataset_functions = [
        MOLECULEACE_DATASET_NAME_TO_LOADER_FUNC[name] for name in dataset_names
    ]

    if as_frames:
        datasets = (
            (dataset_name, load_function(data_dir, as_frame=True, verbose=verbose))
            for dataset_name, load_function in zip(
                dataset_names, dataset_functions, strict=False
            )
        )
    else:
        datasets = (
            (dataset_name, *load_function(data_dir, as_frame=False, verbose=verbose))
            for dataset_name, load_function in zip(
                dataset_names, dataset_functions, strict=False
            )
        )
    return datasets


@validate_params(
    {
        "dataset_name": [StrOptions(set(MOLECULEACE_DATASET_NAMES))],
        "data_dir": [None, str, os.PathLike],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_moleculeace_dataset(
    dataset_name: str,
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str]] | np.ndarray:
    """
    Load MoleculeACE dataset by name.

    Loads a given dataset from MoleculeACE [1]_ benchmark by its name. This is a proxy
    for easier benchmarking, that avoids looking for individual functions.

    Dataset names here are the same as returned by `load_moleculenet_benchmark` function,
    and are case-sensitive.

    Parameters
    ----------
    dataset_name : {}
        Name of the dataset to load.

    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If ``None``, currently set scikit-learn directory
        is used, by default `$HOME/scikit_learn_data`.

    as_frame : bool, default=False
        If True, returns the raw DataFrame with columns "SMILES" and labels
        (dataset-dependent). Otherwise, returns SMILES as list of strings, and
        labels as a NumPy array (shape and type are dataset-dependent).

    verbose : bool, default=False
        If True, progress bar will be shown for downloading or loading files.

    Returns
    -------
    data : pd.DataFrame or tuple(list[str], np.ndarray)
        Depending on the ``as_frame`` argument, one of:
        - Pandas DataFrame with columns depending on the dataset
        - tuple of: list of strings (SMILES), NumPy array (labels)

    References
    ----------
    .. [1] `D. van Tilborg, A. Alenicheva, and F. Grisoni
        “Exposing the Limitations of Molecular Machine Learning with Activity Cliffs”
        Journal of Chemical Information and Modeling, vol. 62, no. 23, pp. 5938–5951, Dec. 2022.
        <https://doi.org/10.1021/acs.jcim.2c01073>`_

    Examples
    --------
    >> from skfp.datasets.moleculeace import load_moleculeace_dataset
    >> dataset = load_moleculeace_dataset("chembl204_ki")
    >> dataset   # doctest: +SKIP
    (['CCCCCCCC(=O)OC[C@H](NC(=O)CN)C(=O)N[C@@H](CO)C(=O)N[C@@H](Cc1ccccc1)C(=O)O', ..., '])
    """
    loader_func = MOLECULEACE_DATASET_NAME_TO_LOADER_FUNC[dataset_name]
    return loader_func(data_dir, as_frame, verbose)


def _subset_to_dataset_names(subset: str | list[str] | None) -> list[str]:
    if subset is None:
        dataset_names = MOLECULEACE_DATASET_NAMES
    elif isinstance(subset, (list, set, tuple)):
        for name in subset:
            if name not in MOLECULEACE_DATASET_NAMES:
                raise ValueError(
                    f"Dataset name '{name}' not recognized among MoleculeACE datasets"
                )
        dataset_names = subset
    else:
        raise ValueError(
            f'Value "{subset}" for subset not recognized, must be a list of strings'
            f"with dataset names from MoleculeACE to load"
        )
    return dataset_names
