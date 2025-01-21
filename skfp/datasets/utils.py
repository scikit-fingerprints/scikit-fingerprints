import json
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from huggingface_hub.utils import (
    are_progress_bars_disabled,
    disable_progress_bars,
    enable_progress_bars,
)
from sklearn.datasets import get_data_home as get_sklearn_data_home


def fetch_dataset(
    data_dir: Optional[Union[str, os.PathLike]],
    dataset_name: str,
    filename: str,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Fetch the dataset from HuggingFace Hub. Returns loaded DataFrame.
    """
    data_home_dir = get_data_home_dir(data_dir, dataset_name)
    dataset_dir = hf_hub_download(data_home_dir, dataset_name, verbose)
    filepath = Path(dataset_dir) / filename
    return pd.read_csv(filepath)


def fetch_splits(
    data_dir: Optional[Union[str, os.PathLike]],
    dataset_name: str,
    filename: str,
    verbose: bool = False,
) -> dict[str, list[int]]:
    """
    Fetch the dataset splits from HuggingFace Hub. Returns loaded JSON with split
    names as keys (e.g. train, valid, test), and lists of indexes as values.
    """
    data_home_dir = get_data_home_dir(data_dir, dataset_name)
    dataset_dir = hf_hub_download(data_home_dir, dataset_name, verbose)
    filepath = Path(dataset_dir) / filename
    if verbose:
        print(filepath)
    with open(filepath) as file:
        return json.load(file)


def get_data_home_dir(
    data_dir: Optional[Union[str, os.PathLike]], dataset_name: str
) -> str:
    """
    Get the data home directory. If valid dataset path is provided, first
    ensures it exists (and all directories in the path). Otherwise, it uses the
    scikit-learn directory, by default `$HOME/scikit_learn_data`.
    """
    if data_dir is None:
        data_dir = Path(get_sklearn_data_home()) / dataset_name
    else:
        data_dir = Path(data_dir) / dataset_name
        data_dir.mkdir(parents=True, exist_ok=True)

    return str(data_dir)


def hf_hub_download(data_home_dir: str, dataset_name: str, verbose: bool) -> str:
    """
    Download the given scikit-fingerprints dataset from HuggingFace Hub.
    Returns the absolute path to the directory with downloaded dataset.
    """
    pbar_was_disabled = are_progress_bars_disabled()
    try:
        if not verbose:
            disable_progress_bars()

        return snapshot_download(
            f"scikit-fingerprints/{dataset_name}",
            repo_type="dataset",
            local_dir=data_home_dir,
            cache_dir=data_home_dir,
        )
    finally:
        if not pbar_was_disabled:
            enable_progress_bars()


def get_mol_strings_and_labels(
    df: pd.DataFrame, mol_type: str = "SMILES"
) -> tuple[list[str], np.ndarray]:
    """
    Extract molecule strings (either SMILES or aminoacid sequences) and labels (one
    or more) from the given DataFrame.

    If ``mol_type`` is ``"SMILES"``, assumes that
    SMILES strings are in the "SMILES" column, another option is ``"aminoseq"``, which
    works similarly, but for aminoacid sequences. All other columns are taken as labels.

    If there is only a single task, labels are returned as a vector.
    """
    if mol_type == "SMILES":
        mol_strings = df.pop("SMILES").tolist()
    elif mol_type == "aminoseq":
        mol_strings = df.pop("aminoseq").tolist()
    else:
        raise ValueError(f"mol_type {mol_type} not recognized")

    # make sure we remove both columns if present
    df = df.drop(columns=["SMILES", "aminoseq"], errors="ignore")

    labels = df.to_numpy()
    if labels.shape[1] == 1:
        labels = labels.ravel()

    return mol_strings, labels
