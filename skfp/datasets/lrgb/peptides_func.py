import os
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.utils._param_validation import StrOptions, validate_params

from skfp.datasets.utils import fetch_dataset, get_mol_strings_and_labels


@validate_params(
    {
        "data_dir": [None, str, os.PathLike],
        "mol_type": [StrOptions({"SMILES", "aminoseq"})],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_peptides_func(
    data_dir: Optional[Union[str, os.PathLike]] = None,
    mol_type: str = "SMILES",
    as_frame: bool = False,
    verbose: bool = False,
) -> Union[pd.DataFrame, tuple[list[str]], np.ndarray]:
    """
    Load and return the Peptides-func dataset.

    The task is to predict functional properties (binary classes) for a set
    of peptides (small proteins) [1]_.

    ==================   =================
    Tasks                               10
    Task type               classification
    Total samples                    15535
    Recommended split    stratified random
    Recommended metric               AUPRC
    ==================   =================

    Parameters
    ----------
    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If ``None``, currently set scikit-learn directory
        is used, by default `$HOME/scikit_learn_data`.

    mol_type : {"SMILES", "aminoseq"}, default="SMILES"
        Which molecule representation to return, either SMILES strings or aminoacid
        sequences.

    as_frame : bool, default=False
        If True, returns the raw DataFrame with columns: "SMILES"/"aminoseq" and "label".
        This depends on the ``mol_type`` parameter. Otherwise, returns molecules as
        a list of strings (either SMILES or aminoacid sequences), and labels as a NumPy
        array (2D integer array).

    verbose : bool, default=False
        If True, progress bar will be shown for downloading or loading files.

    Returns
    -------
    data : pd.DataFrame or tuple(list[str], np.ndarray)
        Depending on the ``as_frame`` and ``mol_type`` parameters, one of:
        - Pandas DataFrame with columns: "SMILES"/"aminoseq", "label"
        - tuple of: list of strings (SMILES / aminoacid sequences), NumPy array (labels)

    References
    ----------
    .. [1] `Dwivedi, Vijay Prakash, et al.
        "Long Range Graph Benchmark"
        Advances in Neural Information Processing Systems 35 (2022): 22326-22340
        <https://proceedings.neurips.cc/paper_files/paper/2022/hash/8c3c666820ea055a77726d66fc7d447f-Abstract-Datasets_and_Benchmarks.html>`_
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="LRGB_Peptides-func",
        filename="peptides_func.csv",
        verbose=verbose,
    )
    return df if as_frame else get_mol_strings_and_labels(df, mol_type=mol_type)
