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
    Load and return the AMES dataset from TDC benchmark [1]_.

    The task is to predict the potential of drugs to induce genetic alterations given a set of molecules [2]_.
    The labels for the data were created with Ames test - a bacterial mutation assay devised by Bruce Ames.

    ==================   =================
    Tasks                                1
    Task type               classification
    Total samples                     7278
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
    -------

    References
    ----------
    .. [1] `Huang, Kexin, et al.
        "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development."
        arXiv preprint arXiv: 2102.09548 (2021)
        <https://arxiv.org/abs/2102.09548>`_

    .. [2] `Xu, Congying, et al.
        “In silico Prediction of Chemical Ames Mutagenicity.”
        Journal of Chemical Information and Modeling 52.11 (2012): 2840-2847
        <https://doi.org/10.1021/ci300400a>`_



    """
    df = fetch_dataset(
        data_dir,
        dataset_name="TDC_ames",
        filename="tdc_ames.csv",
        verbose=verbose,
    )
    return df if as_frame else get_mol_strings_and_labels(df)
