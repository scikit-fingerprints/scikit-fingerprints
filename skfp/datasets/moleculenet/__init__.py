import os
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.utils._param_validation import StrOptions, validate_params

from skfp.datasets.utils import fetch_splits

from .bace import load_bace
from .bbbp import load_bbbp
from .clintox import load_clintox
from .esol import load_esol
from .freesolv import load_freesolv
from .hiv import load_hiv
from .lipophilicity import load_lipophilicity
from .muv import load_muv
from .pcba import load_pcba
from .sider import load_sider
from .tox21 import load_tox21
from .toxcast import load_toxcast


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
    following Open Graph Benchmark [1]_. They differ in recommended metrics. For more
    details, see loading functions for particular datasets.

    Often only a subset of those datasets is used for benchmarking, e.g. only
    single-task datasets, or only classification datasets and excluding PCBA (due to its
    large size). A subset of datasets can be selected by using ``subset`` argument.

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
    Load and return the MoleculeNet dataset splits from Open Graph Benchmark (OGB).

    OGB [1]_ uses precomputed scaffold split with 80/10/10% split between train/valid/test
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
