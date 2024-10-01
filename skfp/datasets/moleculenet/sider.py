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
def load_sider(
    data_dir: Optional[Union[str, os.PathLike]] = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> Union[pd.DataFrame, tuple[list[str]], np.ndarray]:
    """
    Load and return the SIDER (Side Effect Resource) dataset.

    The task is to predict adverse drug reactions (ADRs) as drug side effects to
    27 system organ classes in MedDRA classification. All tasks are binary.

    ==================   ========================
    Tasks                                      27
    Task type            multitask classification
    Total samples                            1427
    Recommended split                    scaffold
    Recommended metric                      AUROC
    ==================   ========================

    Parameters
    ----------
    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If `None`, currently set scikit-learn directory
        is used, by default `$HOME/scikit_learn_data`.

    as_frame : bool, default=False
        If True, returns the raw DataFrame with columns "SMILES" and 27 label columns,
        with names corresponding to MedDRA system organ classes (see [1]_ for details).
        Otherwise, returns SMILES as list of strings,and labels as a NumPy array (2D
        integer array).

    verbose : bool, default=False
        If True, progress bar will be shown for downloading or loading files.

    Returns
    -------
    data : pd.DataFrame or tuple(list[str], np.ndarray)
        Depending on the `as_frame` argument, one of:
        - Pandas DataFrame with columns "SMILES" and 27 label columns
        - tuple of: list of strings (SMILES), NumPy array (labels)

    References
    ----------
    .. [1] `Han Altae-Tran et al.
        "Low Data Drug Discovery with One-Shot Learning"
        ACS Cent. Sci. 2017, 3, 4, 283–293
        <https://pubs.acs.org/doi/10.1021/acscentsci.6b00367>`_

    .. [2] `Zhenqin Wu et al.
        "MoleculeNet: a benchmark for molecular machine learning"
        Chem. Sci., 2018,9, 513-530
        <https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a>`_
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="MoleculeNet_SIDER",
        filename="sider.csv",
        verbose=verbose,
    )
    return df if as_frame else get_smiles_and_labels(df)
