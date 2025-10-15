import os

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
def load_chembl204_ki(
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str]] | np.ndarray:
    """
    Load the ChEMBL204 Ki dataset.

    The task is to predict the inhibitor constant (Ki) of molecules against the Prothrombin target [1]_ [2]_.


    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                  2754
    Recommended split    activity_cliff
    Recommended metric             RMSE
    ==================   ==============

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
    .. [1] `D. van Tilborg, A. Alenicheva, and F. Grisoni
        “Exposing the Limitations of Molecular Machine Learning with Activity Cliffs”
        Journal of Chemical Information and Modeling, vol. 62, no. 23, pp. 5938–5951, Dec. 2022
        <https://doi.org/10.1021/acs.jcim.2c01073>`_

    .. [2] `B. Zdrazil et al.
        “The ChEMBL Database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods”
        Nucleic Acids Research, vol. 52, no. D1, Nov. 2023
        <https://doi.org/10.1093/nar/gkad1004>`_

    Examples
    --------
    >>> from skfp.datasets.moleculenet import load_chembl204_ki
    >>> dataset = load_chembl204_ki()
    >>> dataset  # doctest: +SKIP
    (['CC(=N)N1CCC(Oc2ccc3nc(CCC(=O)O)n(Cc4ccc5ccc(C(=N)N)cc5c4)c3c2)CC1, ..., 'CCC(=O)N1CCC[C@H]1C(=O)NCc1ccc(C(=N)N)cc1'], \
    array([-3.427, ..., -4.146]))

    >>> dataset = load_chembl204_ki(as_frame=True)
    >>> dataset.head() # doctest: +NORMALIZE_WHITESPACE
                                                                        SMILES        Ki
    0         CC(=N)N1CCC(Oc2ccc3nc(CCC(=O)O)n(Cc4ccc5ccc(C(=N)N)cc5c4)c3c2)CC1 -3.426511
    1            CC(=N)N1CCC(Oc2ccc3c(c2)nc(C(C)C)n3Cc2ccc3ccc(C(=N)N)cc3c2)CC1 -2.939519
    2             CCC(C)c1nc2cc(OC3CCN(C(C)=N)CC3)ccc2n1Cc1ccc2ccc(C(=N)N)cc2c1 -3.361728
    3  COC(=O)C(C)CN(c1ccc2c(c1)nc(C)n2Cc1ccc2ccc(C(=N)N)cc2c1)C1CCN(C(C)=N)CC1 -3.698970
    4               CCCCc1nc2cc(OC3CCN(C(C)=N)CC3)ccc2n1Cc1ccc2ccc(C(=N)N)cc2c1 -3.301030
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="MoleculeACE_chembl204_ki",
        filename="chembl204_ki.csv",
        verbose=verbose,
    )
    return df if as_frame else get_mol_strings_and_labels(df)
