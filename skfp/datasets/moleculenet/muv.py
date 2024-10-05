import os
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.utils._param_validation import validate_params

from skfp.datasets.utils import fetch_dataset, get_smiles_and_labels


@validate_params(
    {
        "data_dir": [None, str, os.PathLike],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_muv(
    data_dir: Optional[Union[str, os.PathLike]] = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> Union[pd.DataFrame, tuple[list[str]], np.ndarray]:
    """
    Load and return the MUV (Maximum Unbiased Validation) dataset.

    The task is to predict 17 targets designed for validation of virtual screening
    techniques, based on PubChem BioAssays. All tasks are binary.

    Note that targets have missing values. Algorithms should be evaluated only on
    present labels. For training data, you may want to impute them, e.g. with zeros.

    ==================   ========================
    Tasks                                      17
    Task type            multitask classification
    Total samples                           93087
    Recommended split                    scaffold
    Recommended metric               AUPRC, AUROC
    ==================   ========================

    Parameters
    ----------
    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If ``None``, currently set scikit-learn directory
        is used, by default `$HOME/scikit_learn_data`.

    as_frame : bool, default=False
        If True, returns the raw DataFrame with columns "SMILES" and 17 label columns,
        with names corresponding to MUV targets (see [1]_ and [2]_ for details).
        Otherwise, returns SMILES as list of strings, and labels as a NumPy array.
        Labels are 2D NumPy float array with binary labels and missing values.

    verbose : bool, default=False
        If True, progress bar will be shown for downloading or loading files.

    Returns
    -------
    data : pd.DataFrame or tuple(list[str], np.ndarray)
        Depending on the ``as_frame`` argument, one of:
        - Pandas DataFrame with columns "SMILES" and 17 label columns
        - tuple of: list of strings (SMILES), NumPy array (labels)

    References
    ----------
    .. [1] `Ramsundar, Bharath, et al.
        "Massively multitask networks for drug discovery"
        arXiv:1502.02072 (2015)
        <https://arxiv.org/abs/1502.02072>`_

    .. [2] `Zhenqin Wu et al.
        "MoleculeNet: a benchmark for molecular machine learning"
        Chem. Sci., 2018,9, 513-530
        <https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a>`_
    """
    df = fetch_dataset(
        data_dir, dataset_name="MoleculeNet_MUV", filename="muv.csv", verbose=verbose
    )
    return df if as_frame else get_smiles_and_labels(df)
