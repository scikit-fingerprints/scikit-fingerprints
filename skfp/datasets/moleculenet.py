import os
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.utils._param_validation import StrOptions, validate_params

from skfp.datasets.utils import fetch_dataset, fetch_splits, get_smiles_and_labels


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
        "verbose": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_moleculenet_benchmark(
    subset: Optional[str] = None,
    data_dir: Optional[Union[str, os.PathLike]] = None,
    as_frames: bool = False,
    verbose: bool = False,
) -> Union[list[tuple[str, pd.DataFrame]], list[tuple[str, list[str], np.ndarray]]]:
    """
    Load and return the MoleculeNet benchmark datasets.

    Datasets have varied molecular property prediction tasks: regression, single-task,
    and multitask classification. Scaffold split is recommended for all of them,
    following Open Graph Benchmark [2]_. They differ in recommended metrics. For more
    details, see loading functions for particular datasets.

    Often only a subset of those datasets is used for benchmarking, e.g. only
    single-task datasets, or only classification datasets and excluding PCBA (due to its
    large size). A subset of datasets can be selected by using `subset` argument.

    Dataset names are also returned (case-sensitive). Datasets, grouped by task, are:

    - regression: ESOL, FreeSolv, Lipophilicity
    - single-task classification: BACE, BBBP, HIV
    - multitask classification: ClinTox, MUV, SIDER, Tox21, ToxCast, PCBA

    Parameters
    ----------
    subset : {None, "regression", "classification", "classification_single_task",
              "classification_multitask", "classification_no_pcba"}
        If not None, returns the given subset of datasets.

    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If `None`, currently set scikit-learn directory
        is used, by default `$HOME/scikit_learn_data`.

    as_frames : bool, default=False
        If True, returns the raw DataFrame for each dataset. Otherwise, returns SMILES
        as a list of strings, and labels as a NumPy array for each dataset.

    verbose : bool, default=False
        If True, progress bar will be shown for downloading or loading files.

    Returns
    -------
    data : pd.DataFrame or tuple(list[str], np.ndarray)
        Depending on the `as_frame` argument, one of:
        - Pandas DataFrame with columns: "SMILES", "label"
        - tuple of: list of strings (SMILES), NumPy array (labels)

    References
    ----------
    .. [1] `Zhenqin Wu et al.
        "MoleculeNet: a benchmark for molecular machine learning"
        Chem. Sci., 2018,9, 513-530
        <https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a>`_
    """
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
            (dataset_name, load_function(data_dir, as_frame=True, verbose=verbose))
            for dataset_name, load_function in dataset_functions
        ]
    else:
        # list of tuples (dataset_name, SMILES, y)
        datasets = [
            (dataset_name, *load_function(data_dir, as_frame=False, verbose=verbose))
            for dataset_name, load_function in dataset_functions
        ]

    return datasets


@validate_params(
    {
        "dataset_name": [
            StrOptions(
                {
                    "ESOL",
                    "FreeSolv",
                    "Lipophilicity",
                    "BACE",
                    "BBBP",
                    "HIV",
                    "ClinTox",
                    "MUV",
                    "SIDER",
                    "Tox21",
                    "ToxCast",
                    "PCBA",
                }
            )
        ],
        "data_dir": [None, str, os.PathLike],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_ogb_splits(
    dataset_name: str,
    data_dir: Optional[Union[str, os.PathLike]] = None,
    as_dict: bool = False,
    verbose: bool = False,
) -> Union[tuple[list[int], list[int], list[int]], dict[str, list[int]]]:
    """
    Load and return the MoleculeNet dataset splits from Open Graph Benchmark (OGB) [1]_.

    OGB uses precomputed scaffold split with 80/10/10% split between train/valid/test
    subsets. Test set consists of the smallest scaffold groups, and follows MoleculeNet
    paper [2]_. Those splits are widely used in literature.

    Dataset names here are the same as returned by `load_moleculenet_benchmark` function,
    and are case-sensitive.

    Parameters
    ----------
    dataset_name : {"ESOL", "FreeSolv", "Lipophilicity","BACE", "BBBP", "HIV", "ClinTox",
        "MUV", "SIDER", "Tox21", "ToxCast", "PCBA"}
        Name of the dataset to loads splits for.

    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If `None`, currently set scikit-learn directory
        is used, by default `$HOME/scikit_learn_data`.

    as_dict : bool, default=False
        If True, returns the splits as dictionary with keys "train", "valid" and "test",
        and index lists as values. Otherwise returns three lists with splits indexes.

    verbose : bool, default=False
        If True, progress bar will be shown for downloading or loading files.

    Returns
    -------
    data : tuple(list[int], list[int], list[int]) or dict
        Depending on the `as_dict` argument, one of:
        - three lists of integer indexes
        - dictionary with "train", "valid" and "test" keys, and values as lists with
          splits indexes

    References
    ----------
    .. [1] `Hu, Weihua, et al.
        "Open Graph Benchmark: Datasets for Machine Learning on Graphs."
        Advances in Neural Information Processing Systems 33 (2020): 22118-22133.
        <https://papers.neurips.cc/paper/2020/file/fb60d411a5c5b72b2e7d3527cfc84fd0-Paper.pdf>`_

    .. [2] `Zhenqin Wu et al.
        "MoleculeNet: a benchmark for molecular machine learning"
        Chem. Sci., 2018,9, 513-530
        <https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a>`_
    """
    splits = fetch_splits(
        data_dir,
        dataset_name=f"MoleculeNet_{dataset_name}",
        filename=f"ogb_splits_{dataset_name.lower()}.json",
        verbose=verbose,
    )
    if as_dict:
        return splits
    else:
        return splits["train"], splits["valid"], splits["test"]


@validate_params(
    {
        "data_dir": [None, str, os.PathLike],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_esol(
    data_dir: Optional[Union[str, os.PathLike]] = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> Union[pd.DataFrame, tuple[list[str]], np.ndarray]:
    """
    Load and return the ESOL (Estimated SOLubility) dataset.

    The task is to predict aqueous solubility. Targets are log-transformed,
    and the unit is log mols per litre (log Mol/L).

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                  1128
    Recommended split          scaffold
    Recommended metric             RMSE
    ==================   ==============

    Parameters
    ----------
    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If `None`, currently set scikit-learn directory
        is used, by default `$HOME/scikit_learn_data`.

    as_frame : bool, default=False
        If True, returns the raw DataFrame with columns: "SMILES", "label". Otherwise,
        returns SMILES as list of strings, and labels as a NumPy array (1D float vector).

    verbose : bool, default=False
        If True, progress bar will be shown for downloading or loading files.

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
    df = fetch_dataset(
        data_dir,
        dataset_name="MoleculeNet_ESOL",
        filename="esol.csv",
        verbose=verbose,
    )
    return df if as_frame else get_smiles_and_labels(df)


@validate_params(
    {
        "data_dir": [None, str, os.PathLike],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_freesolv(
    data_dir: Optional[Union[str, os.PathLike]] = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> Union[pd.DataFrame, tuple[list[str]], np.ndarray]:
    """
    Load and return the FreeSolv (Free Solvation Database) dataset.

    The task is to predict hydration free energy of small molecules in water.
    Targets are in kcal/mol.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                   642
    Recommended split          scaffold
    Recommended metric             RMSE
    ==================   ==============

    Parameters
    ----------
    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If `None`, currently set scikit-learn directory
        is used, by default `$HOME/scikit_learn_data`.

    as_frame : bool, default=False
        If True, returns the raw DataFrame with columns: "SMILES", "label". Otherwise,
        returns SMILES as list of strings, and labels as a NumPy array (1D float vector).

    verbose : bool, default=False
        If True, progress bar will be shown for downloading or loading files.

    Returns
    -------
    data : pd.DataFrame or tuple(list[str], np.ndarray)
        Depending on the `as_frame` argument, one of:
        - Pandas DataFrame with columns: "SMILES", "label"
        - tuple of: list of strings (SMILES), NumPy array (labels)

    References
    ----------
    .. [1] `Mobley, D.L., Guthrie, J.P.
        "FreeSolv: a database of experimental and calculated hydration free energies,
        with input files"
        J Comput Aided Mol Des 28, 711–720 (2014)
        <https://link.springer.com/article/10.1007/s10822-014-9747-x>`_

    .. [2] `Zhenqin Wu et al.
        "MoleculeNet: a benchmark for molecular machine learning"
        Chem. Sci., 2018,9, 513-530
        <https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a>`_
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="MoleculeNet_FreeSolv",
        filename="freesolv.csv",
        verbose=verbose,
    )
    return df if as_frame else get_smiles_and_labels(df)


@validate_params(
    {
        "data_dir": [None, str, os.PathLike],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_lipophilicity(
    data_dir: Optional[Union[str, os.PathLike]] = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> Union[pd.DataFrame, tuple[list[str]], np.ndarray]:
    """
    Load and return the Lipophilicity dataset.

    The task is to predict octanol/water distribution coefficient (logD) at pH 7.4.
    Targets are already log transformed, and are a unitless ratio.

    ==================   ==============
    Tasks                             1
    Task type                regression
    Total samples                  4200
    Recommended split          scaffold
    Recommended metric             RMSE
    ==================   ==============

    Parameters
    ----------
    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If `None`, currently set scikit-learn directory
        is used, by default `$HOME/scikit_learn_data`.

    as_frame : bool, default=False
        If True, returns the raw DataFrame with columns: "SMILES", "label". Otherwise,
        returns SMILES as list of strings, and labels as a NumPy array (1D float vector).

    verbose : bool, default=False
        If True, progress bar will be shown for downloading or loading files.

    Returns
    -------
    data : pd.DataFrame or tuple(list[str], np.ndarray)
        Depending on the `as_frame` argument, one of:
        - Pandas DataFrame with columns: "SMILES", "label"
        - tuple of: list of strings (SMILES), NumPy array (labels)

    References
    ----------
    .. [1] `Zhenqin Wu et al.
        "MoleculeNet: a benchmark for molecular machine learning"
        Chem. Sci., 2018,9, 513-530
        <https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a>`_
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="MoleculeNet_Lipophilicity",
        filename="lipophilicity.csv",
        verbose=verbose,
    )
    return df if as_frame else get_smiles_and_labels(df)


@validate_params(
    {
        "data_dir": [None, str, os.PathLike],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_bace(
    data_dir: Optional[Union[str, os.PathLike]] = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> Union[pd.DataFrame, tuple[list[str]], np.ndarray]:
    """
    Load and return the BACE dataset.

    The task is to predict binding results for a set of inhibitors of human
    β-secretase 1 (BACE-1).

    ==================   ==============
    Tasks                             1
    Task type            classification
    Total samples                  1513
    Recommended split          scaffold
    Recommended metric            AUROC
    ==================   ==============

    Parameters
    ----------
    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If `None`, currently set scikit-learn directory
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
        Depending on the `as_frame` argument, one of:
        - Pandas DataFrame with columns: "SMILES", "label"
        - tuple of: list of strings (SMILES), NumPy array (labels)

    References
    ----------
    .. [1] `Govindan Subramanian et al.
        "Computational Modeling of β-Secretase 1 (BACE-1) Inhibitors Using
        Ligand Based Approaches"
        J. Chem. Inf. Model. 2016, 56, 10, 1936–1949
        <https://pubs.acs.org/doi/10.1021/acs.jcim.6b00290>`_

    .. [2] `Zhenqin Wu et al.
        "MoleculeNet: a benchmark for molecular machine learning"
        Chem. Sci., 2018,9, 513-530
        <https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a>`_
    """
    df = fetch_dataset(
        data_dir, dataset_name="MoleculeNet_BACE", filename="bace.csv", verbose=verbose
    )
    return df if as_frame else get_smiles_and_labels(df)


@validate_params(
    {
        "data_dir": [None, str, os.PathLike],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_bbbp(
    data_dir: Optional[Union[str, os.PathLike]] = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> Union[pd.DataFrame, tuple[list[str]], np.ndarray]:
    """
    Load and return the BBBP (Blood-Brain Barrier Penetration) dataset.

    The task is to predict blood-brain barrier penetration (barrier permeability)
    of small drug-like molecules.

    ==================   ==============
    Tasks                             1
    Task type            classification
    Total samples                  2039
    Recommended split          scaffold
    Recommended metric            AUROC
    ==================   ==============

    Parameters
    ----------
    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If `None`, currently set scikit-learn directory
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
        Depending on the `as_frame` argument, one of:
        - Pandas DataFrame with columns: "SMILES", "label"
        - tuple of: list of strings (SMILES), NumPy array (labels)

    References
    ----------
    .. [1] `Ines Filipa Martins et al.
        "A Bayesian Approach to in Silico Blood-Brain Barrier Penetration Modeling"
        J. Chem. Inf. Model. 2012, 52, 6, 1686–1697
        <https://pubs.acs.org/doi/10.1021/ci300124c>`_

    .. [2] `Zhenqin Wu et al.
        "MoleculeNet: a benchmark for molecular machine learning"
        Chem. Sci., 2018,9, 513-530
        <https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a>`_
    """
    df = fetch_dataset(
        data_dir, dataset_name="MoleculeNet_BBBP", filename="bbbp.csv", verbose=verbose
    )
    return df if as_frame else get_smiles_and_labels(df)


@validate_params(
    {
        "data_dir": [None, str, os.PathLike],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_hiv(
    data_dir: Optional[Union[str, os.PathLike]] = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> Union[pd.DataFrame, tuple[list[str]], np.ndarray]:
    """
    Load and return the HIV dataset.

    The task is to predict ability of molecules to inhibit HIV replication.

    ==================   ==============
    Tasks                             1
    Task type            classification
    Total samples                 41127
    Recommended split          scaffold
    Recommended metric            AUROC
    ==================   ==============

    Parameters
    ----------
    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If `None`, currently set scikit-learn directory
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
        Depending on the `as_frame` argument, one of:
        - Pandas DataFrame with columns: "SMILES", "label"
        - tuple of: list of strings (SMILES), NumPy array (labels)

    References
    ----------
    .. [1] `AIDS Antiviral Screen Data
        <https://wiki.nci.nih.gov/display/NCIDTPdata/AIDS+Antiviral+Screen+Data>`_

    .. [2] `Zhenqin Wu et al.
        "MoleculeNet: a benchmark for molecular machine learning"
        Chem. Sci., 2018,9, 513-530
        <https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a>`_
    """
    df = fetch_dataset(
        data_dir, dataset_name="MoleculeNet_HIV", filename="hiv.csv", verbose=verbose
    )
    return df if as_frame else get_smiles_and_labels(df)


@validate_params(
    {
        "data_dir": [None, str, os.PathLike],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_clintox(
    data_dir: Optional[Union[str, os.PathLike]] = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> Union[pd.DataFrame, tuple[list[str]], np.ndarray]:
    """
    Load and return the ClinTox dataset.

    The task is to predict drug approval viability, by predicting clinical trial
    toxicity and final FDA approval status. Both tasks are binary.

    ==================   ========================
    Tasks                                       2
    Task type            multitask classification
    Total samples                            1477
    Recommended split                    scaffold
    Recommended metric                      AUROC
    ==================   ========================

    Parameters
    ----------
    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If `None`, currently set scikit-learn directory
        is used, by default `$HOME/scikit_learn_data`.

    as_frame : bool, default=False
        If True, returns the raw DataFrame with columns "SMILES" and 2 label columns,
        FDA approval and clinical trial toxicity. Otherwise, returns SMILES as list
        of strings,and labels as a NumPy array (2D integer array).

    verbose : bool, default=False
        If True, progress bar will be shown for downloading or loading files.

    Returns
    -------
    data : pd.DataFrame or tuple(list[str], np.ndarray)
        Depending on the `as_frame` argument, one of:
        - Pandas DataFrame with columns "SMILES" and 2 label columns
        - tuple of: list of strings (SMILES), NumPy array (labels)

    References
    ----------
    .. [1] `Zhenqin Wu et al.
        "MoleculeNet: a benchmark for molecular machine learning"
        Chem. Sci., 2018,9, 513-530
        <https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a>`_
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="MoleculeNet_ClinTox",
        filename="clintox.csv",
        verbose=verbose,
    )
    return df if as_frame else get_smiles_and_labels(df)


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
        Path to the root data directory. If `None`, currently set scikit-learn directory
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
        Depending on the `as_frame` argument, one of:
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


@validate_params(
    {
        "data_dir": [None, str, os.PathLike],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_tox21(
    data_dir: Optional[Union[str, os.PathLike]] = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> Union[pd.DataFrame, tuple[list[str]], np.ndarray]:
    """
    Load and return the Tox21 dataset.

    The task is to predict 12 toxicity targets, including nuclear receptors and
    stress response pathways. All tasks are binary.

    Note that targets have missing values. Algorithms should be evaluated only on
    present labels. For training data, you may want to impute them, e.g. with zeros.

    ==================   ========================
    Tasks                                      12
    Task type            multitask classification
    Total samples                            7831
    Recommended split                    scaffold
    Recommended metric                      AUROC
    ==================   ========================

    Parameters
    ----------
    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If `None`, currently set scikit-learn directory
        is used, by default `$HOME/scikit_learn_data`.

    as_frame : bool, default=False
        If True, returns the raw DataFrame with columns "SMILES" and 12 label columns,
        with names corresponding to toxicity targets (see [1]_ and [2]_ for details).
        Otherwise, returns SMILES as list of strings, and labels as a NumPy array.
        Labels are 2D NumPy float array with binary labels and missing values.

    verbose : bool, default=False
        If True, progress bar will be shown for downloading or loading files.

    Returns
    -------
    data : pd.DataFrame or tuple(list[str], np.ndarray)
        Depending on the `as_frame` argument, one of:
        - Pandas DataFrame with columns "SMILES" and 12 label columns
        - tuple of: list of strings (SMILES), NumPy array (labels)

    References
    ----------
    .. [1] `Tox21 Challenge
        <https://tripod.nih.gov/tox21/challenge/>`_

    .. [2] `Zhenqin Wu et al.
        "MoleculeNet: a benchmark for molecular machine learning"
        Chem. Sci., 2018,9, 513-530
        <https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a>`_
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="MoleculeNet_Tox21",
        filename="tox21.csv",
        verbose=verbose,
    )
    return df if as_frame else get_smiles_and_labels(df)


@validate_params(
    {
        "data_dir": [None, str, os.PathLike],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_toxcast(
    data_dir: Optional[Union[str, os.PathLike]] = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> Union[pd.DataFrame, tuple[list[str]], np.ndarray]:
    """
    Load and return the ToxCast dataset.

    The task is to predict 617 toxicity targets from a large library of compounds
    based on in vitro high-throughput screening. All tasks are binary.

    Note that targets have missing values. Algorithms should be evaluated only on
    present labels. For training data, you may want to impute them, e.g. with zeros.

    ==================   ========================
    Tasks                                     617
    Task type            multitask classification
    Total samples                            8576
    Recommended split                    scaffold
    Recommended metric                      AUROC
    ==================   ========================

    Parameters
    ----------
    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If `None`, currently set scikit-learn directory
        is used, by default `$HOME/scikit_learn_data`.

    as_frame : bool, default=False
        If True, returns the raw DataFrame with columns "SMILES" and 617 label columns,
        with names corresponding to toxicity targets (see [1]_ and [2]_ for details).
        Otherwise, returns SMILES as list of strings, and labels as a NumPy array.
        Labels are 2D NumPy float array with binary labels and missing values.

    verbose : bool, default=False
        If True, progress bar will be shown for downloading or loading files.

    Returns
    -------
    data : pd.DataFrame or tuple(list[str], np.ndarray)
        Depending on the `as_frame` argument, one of:
        - Pandas DataFrame with columns "SMILES" and 617 label columns
        - tuple of: list of strings (SMILES), NumPy array (labels)

    References
    ----------
    .. [1] `Ann M. Richard et al.
        "ToxCast Chemical Landscape: Paving the Road to 21st Century Toxicology"
        Chem. Res. Toxicol. 2016, 29, 8, 1225–1251
        <https://pubs.acs.org/doi/10.1021/acs.chemrestox.6b00135>`_

    .. [2] `Zhenqin Wu et al.
        "MoleculeNet: a benchmark for molecular machine learning"
        Chem. Sci., 2018,9, 513-530
        <https://pubs.rsc.org/en/content/articlelanding/2018/sc/c7sc02664a>`_
    """
    df = fetch_dataset(
        data_dir,
        dataset_name="MoleculeNet_ToxCast",
        filename="toxcast.csv",
        verbose=verbose,
    )
    return df if as_frame else get_smiles_and_labels(df)


@validate_params(
    {
        "data_dir": [None, str, os.PathLike],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_pcba(
    data_dir: Optional[Union[str, os.PathLike]] = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> Union[pd.DataFrame, tuple[list[str]], np.ndarray]:
    """
    Load and return the PCBA (PubChem BioAssay) dataset.

    The task is to predict biological activity against 128 bioassays, generated
    by high-throughput screening (HTS). All tasks are binary active/non-active.

    Note that targets have missing values. Algorithms should be evaluated only on
    present labels. For training data, you may want to impute them, e.g. with zeros.

    ==================   ========================
    Tasks                                     128
    Task type            multitask classification
    Total samples                          437929
    Recommended split                    scaffold
    Recommended metric               AUPRC, AUROC
    ==================   ========================

    Parameters
    ----------
    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If `None`, currently set scikit-learn directory
        is used, by default `$HOME/scikit_learn_data`.

    as_frame : bool, default=False
        If True, returns the raw DataFrame with columns "SMILES" and 128 label columns,
        with names corresponding to biological activities (see [1]_ and [2]_ for details).
        Otherwise, returns SMILES as list of strings, and labels as a NumPy array.
        Labels are 2D NumPy float array with binary labels and missing values.


    verbose : bool, default=False
        If True, progress bar will be shown for downloading or loading files.

    Returns
    -------
    data : pd.DataFrame or tuple(list[str], np.ndarray)
        Depending on the `as_frame` argument, one of:
        - Pandas DataFrame with columns "SMILES" and 128 label columns
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
        data_dir, dataset_name="MoleculeNet_PCBA", filename="pcba.csv", verbose=verbose
    )
    return df if as_frame else get_smiles_and_labels(df)
