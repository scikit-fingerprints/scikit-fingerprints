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
def load_carcinogens_lagunin(
    data_dir: Optional[Union[str, os.PathLike]] = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> Union[pd.DataFrame, tuple[list[str]], np.ndarray]:
    """
    Load and return the Carcinogens dataset.

    The task is to predict whether the drug is a carcinogen [1]_ [2]_ [3]_.

    ==================   =================
    Tasks                                1
    Task type               classification
    Total samples                      280
    Recommended split             scaffold
    Recommended metric               AUROC
    ==================   =================

    Parameters
    ----------
    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If ``None``, currently set scikit-learn directory
        is used, by default `$HOME/scikit_learn_data`.

    as_frame : bool, default=False
        If True, returns the raw DataFrame with columns: "SMILES", "label". Otherwise,
        returns SMILES as list of strings, and labels as a NumPy array (1D integer binary
        vector).

    verbose : bool, default=False
        If True, progress bar will be shown for downloading or loading files.

    Returns
    -------
    data : pd.DataFrame or tuple(list[str], np.ndarray)
        Depending on the ``as_frame`` argument, one of:
        - Pandas DataFrame with columns: "SMILES", "label"
        - tuple of: list of strings (SMILES), NumPy array (labels)

    References
    ----------
    .. [1] `Lagunin, Alexey, et al.
        “Computer-Aided Prediction of Rodent Carcinogenicity by PASS and CISOC-PSCT.”
        QSAR & Combinatorial Science 28.8 (2009): 806-810
        <https://doi.org/10.1002/qsar.200860192>`_

    .. [2] `Cheng, Feixiong, et al.
        “admetSAR: A Comprehensive Source and Free Tool for Assessment of Chemical ADMET Properties.”
        Journal of Chemical Information and Modeling 52.11 (2012): 2840-2847
        <https://doi.org/10.1021/ci300367a>`_

    .. [3] `Huang, Kexin, et al.
        "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development"
        Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks, 2021
        <https://openreview.net/forum?id=8nvgnORnoWr>`_
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="TDC_carcinogens_lagunin",
        filename="tdc_carcinogens_lagunin.csv",
        verbose=verbose,
    )
    return df if as_frame else get_mol_strings_and_labels(df)
