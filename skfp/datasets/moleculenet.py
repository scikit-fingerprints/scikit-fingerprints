import os
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.utils._param_validation import StrOptions, validate_params

from skfp.datasets.utils import fetch_dataset, get_smiles_and_labels


@validate_params(
    {
        "subset": [
            None,
            StrOptions(
                {
                    "classification",
                    "classification_single_task",
                    "classification_multitask",
                    "classification_no_pcba",
                    "regression",
                }
            ),
        ],
        "data_dir": [None, str, os.PathLike],
        "as_frame": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_moleculenet_benchmark(
    subset: Optional[str] = None,
    data_dir: Optional[Union[str, os.PathLike]] = None,
    as_frames: bool = False,
) -> Union[list[tuple[str, pd.DataFrame]], list[tuple[str, list[str], np.ndarray]]]:
    regression_datasets = [
        ("ESOL", load_esol),
        ("FreeSolv", load_freesolv),
        ("Lipophilicity", load_lipophilicity),
    ]
    clf_single_task_datasets = [
        ("BACE", load_bace),
        ("BBBP", load_bbbp),
        ("HIV", load_hiv),
    ]
    clf_multitask_datasets = [
        ("ClinTox", load_clintox),
        ("MUV", load_muv),
        ("SIDER", load_sider),
        ("Tox21", load_tox21),
        ("ToxCast", load_toxcast),
    ]
    clf_pcba = [("PCBA", load_pcba)]

    if subset is None:
        dataset_functions = (
            regression_datasets
            + clf_single_task_datasets
            + clf_multitask_datasets
            + clf_pcba
        )
    elif subset == "classification":
        dataset_functions = clf_single_task_datasets + clf_multitask_datasets + clf_pcba
    elif subset == "classification_single_task":
        dataset_functions = clf_single_task_datasets
    elif subset == "classification_multitask":
        dataset_functions = clf_multitask_datasets
    elif subset == "classification_no_pcba":
        dataset_functions = clf_single_task_datasets + clf_multitask_datasets
    elif subset == "regression":
        dataset_functions = regression_datasets
    else:
        raise ValueError(
            f'Value "{subset}" for subset not recognized, must be one of: '
            f'"classification", "classification_single_task", '
            f'"classification_no_pcba", "regression"'
        )

    if as_frames:
        # list of tuples (dataset_name, DataFrame)
        datasets = [
            (dataset_name, load_function(data_dir, as_frame=True))
            for dataset_name, load_function in dataset_functions
        ]
    else:
        # list of tuples (dataset_name, SMILES, y)
        datasets = [
            (dataset_name, *load_function(data_dir, as_frame=False))
            for dataset_name, load_function in dataset_functions
        ]

    return datasets


@validate_params(
    {"data_dir": [None, str, os.PathLike], "as_frame": ["boolean"]},
    prefer_skip_nested_validation=True,
)
def load_esol(
    data_dir: Optional[Union[str, os.PathLike]] = None,
    as_frame: bool = False,
) -> Union[pd.DataFrame, tuple[list[str]], np.ndarray]:
    """
    Load and return the ESOL (Estimated SOLubility) [1]_ dataset.

    The task is to predict aqueous solubility. Targets are log-transformed,
    and the unit is log mols per litre (log Mol/L).

    In MoleculeNet benchmark [2]_, it belongs to the Physical Chemistry category.

    =================   ==============
    Tasks                            1
    Task type               regression
    Total samples                 1128
    Recommended split         scaffold
    =================   ==============

    Parameters
    ----------
    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If `None`, currently set scikit-learn directory
        is used, by default `$HOME/scikit_learn_data`.

    as_frame : bool, default=False
        If True, returns the raw DataFrame with columns: "SMILES", "label". Otherwise,
        returns SMILES as list of strings, and labels as a NumPy array (1D float vector).

    Returns
    -------
    data : pd.DataFrame or tuple(list[str], np.ndarray)
        Depending on the `as_frame` argument, one of:
        - Pandas DataFrame with columns: "SMILES", "label"
        - tuple of: list of strings (SMILES), NumPy array (labels)

    References
    ----------
    .. [1] `John S. Delaney
        "ESOL: Estimating Aqueous Solubility Directly from Molecular Structure"
        J. Chem. Inf. Comput. Sci. 2004, 44, 3, 1000–1005
        <https://pubs.acs.org/doi/10.1021/ci034243x>`_

    .. [2] `Zhenqin Wu et al.
        "MoleculeNet: a benchmark for molecular machine learning"
        Chem. Sci., 2018,9, 513-530
        <https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a>`_
    """
    df = fetch_dataset(data_dir, dataset_name="MoleculeNet_ESOL", filename="esol.csv")
    return df if as_frame else get_smiles_and_labels(df)


@validate_params(
    {"data_dir": [None, str, os.PathLike], "as_frame": ["boolean"]},
    prefer_skip_nested_validation=True,
)
def load_freesolv(
    data_dir: Optional[Union[str, os.PathLike]] = None,
    as_frame: bool = False,
) -> Union[pd.DataFrame, tuple[list[str]], np.ndarray]:
    """
    Load and return the FreeSolv (Free Solvation Database) [1]_ dataset.

    The task is to predict hydration free energy of small molecules in water.

    The task is to predict aqueous solubility. Targets are log-transformed,
    and the unit is log mols per litre (log Mol/L).

    =================   ==============
    Tasks                            1
    Task type               regression
    Total samples                 1128
    Recommended split         scaffold
    =================   ==============

    Parameters
    ----------
    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If `None`, currently set scikit-learn directory
        is used, by default `$HOME/scikit_learn_data`.

    as_frame : bool, default=False
        If True, returns the raw DataFrame with columns: "SMILES", "label". Otherwise,
        returns SMILES as list of strings, and labels as a NumPy array (1D float vector).

    Returns
    -------
    data : pd.DataFrame or tuple(list[str], np.ndarray)
        Depending on the `as_frame` argument, one of:
        - Pandas DataFrame with columns: "SMILES", "label"
        - tuple of: list of strings (SMILES), NumPy array (labels)

    References
    ----------
    .. [1] `John S. Delaney
        "ESOL: Estimating Aqueous Solubility Directly from Molecular Structure"
        J. Chem. Inf. Comput. Sci. 2004, 44, 3, 1000–1005
        <https://pubs.acs.org/doi/10.1021/ci034243x>`_
    """
    df = fetch_dataset(
        data_dir, dataset_name="MoleculeNet_FreeSolv", filename="freesolv.csv"
    )
    return df if as_frame else get_smiles_and_labels(df)


@validate_params(
    {"data_dir": [None, str, os.PathLike], "as_frame": ["boolean"]},
    prefer_skip_nested_validation=True,
)
def load_lipophilicity(
    data_dir: Optional[Union[str, os.PathLike]] = None,
    as_frame: bool = False,
) -> Union[pd.DataFrame, tuple[list[str]], np.ndarray]:
    df = fetch_dataset(
        data_dir, dataset_name="MoleculeNet_Lipophilicity", filename="lipophilicity.csv"
    )
    return df if as_frame else get_smiles_and_labels(df)


@validate_params(
    {"data_dir": [None, str, os.PathLike], "as_frame": ["boolean"]},
    prefer_skip_nested_validation=True,
)
def load_pcba(
    data_dir: Optional[Union[str, os.PathLike]] = None,
    as_frame: bool = False,
) -> Union[pd.DataFrame, tuple[list[str]], np.ndarray]:
    df = fetch_dataset(data_dir, dataset_name="MoleculeNet_PCBA", filename="pcba.csv")
    return df if as_frame else get_smiles_and_labels(df)


@validate_params(
    {"data_dir": [None, str, os.PathLike], "as_frame": ["boolean"]},
    prefer_skip_nested_validation=True,
)
def load_muv(
    data_dir: Optional[Union[str, os.PathLike]] = None,
    as_frame: bool = False,
) -> Union[pd.DataFrame, tuple[list[str]], np.ndarray]:
    df = fetch_dataset(data_dir, dataset_name="MoleculeNet_MUV", filename="muv.csv")
    return df if as_frame else get_smiles_and_labels(df)


@validate_params(
    {"data_dir": [None, str, os.PathLike], "as_frame": ["boolean"]},
    prefer_skip_nested_validation=True,
)
def load_hiv(
    data_dir: Optional[Union[str, os.PathLike]] = None,
    as_frame: bool = False,
) -> Union[pd.DataFrame, tuple[list[str]], np.ndarray]:
    df = fetch_dataset(data_dir, dataset_name="MoleculeNet_HIV", filename="hiv.csv")
    return df if as_frame else get_smiles_and_labels(df)


@validate_params(
    {"data_dir": [None, str, os.PathLike], "as_frame": ["boolean"]},
    prefer_skip_nested_validation=True,
)
def load_bace(
    data_dir: Optional[Union[str, os.PathLike]] = None,
    as_frame: bool = False,
) -> Union[pd.DataFrame, tuple[list[str]], np.ndarray]:
    df = fetch_dataset(data_dir, dataset_name="MoleculeNet_BACE", filename="bace.csv")
    return df if as_frame else get_smiles_and_labels(df)


@validate_params(
    {"data_dir": [None, str, os.PathLike], "as_frame": ["boolean"]},
    prefer_skip_nested_validation=True,
)
def load_bbbp(
    data_dir: Optional[Union[str, os.PathLike]] = None,
    as_frame: bool = False,
) -> Union[pd.DataFrame, tuple[list[str]], np.ndarray]:
    df = fetch_dataset(data_dir, dataset_name="MoleculeNet_BBBP", filename="bbbp.csv")
    return df if as_frame else get_smiles_and_labels(df)


@validate_params(
    {"data_dir": [None, str, os.PathLike], "as_frame": ["boolean"]},
    prefer_skip_nested_validation=True,
)
def load_tox21(
    data_dir: Optional[Union[str, os.PathLike]] = None,
    as_frame: bool = False,
) -> Union[pd.DataFrame, tuple[list[str]], np.ndarray]:
    df = fetch_dataset(data_dir, dataset_name="MoleculeNet_Tox21", filename="tox21.csv")
    return df if as_frame else get_smiles_and_labels(df)


@validate_params(
    {"data_dir": [None, str, os.PathLike], "as_frame": ["boolean"]},
    prefer_skip_nested_validation=True,
)
def load_toxcast(
    data_dir: Optional[Union[str, os.PathLike]] = None,
    as_frame: bool = False,
) -> Union[pd.DataFrame, tuple[list[str]], np.ndarray]:
    df = fetch_dataset(
        data_dir, dataset_name="MoleculeNet_ToxCast", filename="toxcast.csv"
    )
    return df if as_frame else get_smiles_and_labels(df)


@validate_params(
    {"data_dir": [None, str, os.PathLike], "as_frame": ["boolean"]},
    prefer_skip_nested_validation=True,
)
def load_sider(
    data_dir: Optional[Union[str, os.PathLike]] = None,
    as_frame: bool = False,
) -> Union[pd.DataFrame, tuple[list[str]], np.ndarray]:
    df = fetch_dataset(data_dir, dataset_name="MoleculeNet_SIDER", filename="sider.csv")
    return df if as_frame else get_smiles_and_labels(df)


@validate_params(
    {"data_dir": [None, str, os.PathLike], "as_frame": ["boolean"]},
    prefer_skip_nested_validation=True,
)
def load_clintox(
    data_dir: Optional[Union[str, os.PathLike]] = None,
    as_frame: bool = False,
) -> Union[pd.DataFrame, tuple[list[str]], np.ndarray]:
    df = fetch_dataset(
        data_dir, dataset_name="MoleculeNet_ClinTox", filename="clintox.csv"
    )
    return df if as_frame else get_smiles_and_labels(df)
