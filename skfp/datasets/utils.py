import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download
from sklearn.datasets import get_data_home as get_sklearn_data_home


def fetch_dataset(
    data_dir: Optional[Union[str, os.PathLike]], dataset_name: str, filename: str
) -> pd.DataFrame:
    """
    Fetches the dataset from HuggingFace Hub and returns loaded DataFrame.
    """
    data_home_dir = get_data_home_dir(data_dir, dataset_name)
    dataset_dir = hf_hub_download(data_home_dir, dataset_name)
    return pd.read_csv(Path(dataset_dir) / filename)


def get_data_home_dir(
    data_dir: Optional[Union[str, os.PathLike]], dataset_name: str
) -> str:
    """
    Returns the data home directory. If valid dataset path is provided, first
    ensures it exists (and all directories in the path). Otherwise, it uses the
    scikit-learn directory, by default `$HOME/scikit_learn_data`.
    """
    if data_dir is None:
        data_dir = Path(get_sklearn_data_home()) / dataset_name
    else:
        Path(data_dir).mkdir(parents=True, exist_ok=True)

    return str(data_dir)


def hf_hub_download(data_home_dir: str, dataset_name: str) -> str:
    """
    Downloads the given scikit-fingerprints dataset from HuggingFace Hub.
    Returns the absolute path to the directory with downloaded dataset.
    """
    return snapshot_download(
        f"scikit-fingerprints/{dataset_name}",
        repo_type="dataset",
        local_dir=data_home_dir,
    )


def get_smiles_and_labels(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """
    Extracts SMILES and labels (one or more) from the given DataFrame. Assumes
    that SMILES strings are in "SMILES" column, and all other columns are labels.
    """
    smiles = df.pop("SMILES").tolist()
    labels = df.values
    return smiles, labels
