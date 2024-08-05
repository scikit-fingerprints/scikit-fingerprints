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


def get_smiles_and_labels(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """
    Extract SMILES and labels (one or more) from the given DataFrame. Assumes
    that SMILES strings are in "SMILES" column, and all other columns are labels.
    If there is only a single task, labels are returned as a vector.
    """
    smiles = df.pop("SMILES").tolist()
    labels = df.values
    if labels.shape[1] == 1:
        labels = labels.ravel()
    return smiles, labels
