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
def load_dili(
    data_dir: Optional[Union[str, os.PathLike]] = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> Union[pd.DataFrame, tuple[list[str]], np.ndarray]:
    """
    Load and return the DILI (Drug Induced Liver Injury) dataset from TDC benchmark [1]_.

    The task is to predict whether the drugs can cause liver injury [2]_.
    Drug caused liver injuries have been recognized as the single,
    most frequent cause of safety-related drug marketing withdrawal.
    This dataset is aggregated from U.S. FDA’s National Center for Toxicological Research.


    ==================   =================
    Tasks                                1
    Task type               classification
    Total samples                      475
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

    .. [2] `Xu, Youjun, et al.
        “Deep Learning for Drug-Induced Liver Injury.”
        Journal of Chemical Information and Modeling 55.10 (2015): 2085-2093
        <https://doi.org/10.1021/acs.jcim.5b00238>`_
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="TDC_dili",
        filename="tdc_dili.csv",
        verbose=verbose,
    )
    return df if as_frame else get_mol_strings_and_labels(df)
