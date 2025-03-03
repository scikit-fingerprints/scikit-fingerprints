import os
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.utils._param_validation import validate_params

from skfp.datasets.utils import fetch_dataset, get_mol_strings_and_labels


@validate_params(
    {
        "data_dir": [None, str, os.PathLike],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_ames(
    data_dir: Optional[Union[str, os.PathLike]] = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> Union[pd.DataFrame, tuple[list[str]], np.ndarray]:
    """
    Load and return the AMES dataset from TDC benchmark.

    The task is to predict the ability of drugs to cause
    genetic mutations given a set of molecules [1]_.

    ==================   =================
    Tasks                                1
    Task type               classification
    Total samples                     7278
    Recommended split      scaffold/random
    Recommended metric               AUROC
    ==================   =================

    Parameters
    ----------
    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If ``None``, currently set scikit-learn directory
        is used, by default `$HOME/scikit_learn_data`.

    as_frame : bool, default=False
        If True, returns the raw DataFrame with columns: "SMILES", "Y". Otherwise,
        returns SMILES as list of strings, and labels as a NumPy array (1D integer binary
        vector).

    verbose : bool, default=False
        If True, progress bar will be shown for downloading or loading files.

    Returns
    -------
    data : pd.DataFrame or tuple(list[str], np.ndarray)
        Depending on the ``as_frame`` argument, one of:
        - Pandas DataFrame with columns: "SMILES", "Y"
        - tuple of: list of strings (SMILES), NumPy array (labels)
    -------

    References
    ----------
    .. [1] `Xu, Congying, et al.
        “In silico prediction of chemical Ames mutagenicity.”
        Journal of chemical information and modeling 52.11 (2012): 2840-2847
        <https://doi.org/10.1021/ci300400a>`_

    """
    df = fetch_dataset(
        data_dir,
        dataset_name="TDC_ames",
        filename="tdc_ames.csv",
        verbose=verbose,
    )
    return df if as_frame else get_mol_strings_and_labels(df)
