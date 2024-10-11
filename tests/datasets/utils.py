import os
import shutil

import numpy as np
import pandas as pd

from skfp.datasets.utils import get_data_home_dir, get_mol_strings_and_labels


def test_get_data_home_dir():
    default_dir = get_data_home_dir(None, dataset_name="test")
    assert default_dir.endswith(os.path.join("scikit_learn_data", "test"))

    try:
        custom_dir = get_data_home_dir(data_dir="./data", dataset_name="test")
    finally:
        shutil.rmtree("data")
    assert custom_dir == os.path.join("data", "test")


def test_get_smiles_and_labels():
    df = pd.DataFrame({"SMILES": ["a", "b", "c"], "label": [0, 0, 1]})
    smiles_list, y = get_mol_strings_and_labels(df)
    assert smiles_list == ["a", "b", "c"]
    assert y.ndim == 1
    assert np.array_equal(y, np.array([0, 0, 1]))

    df = pd.DataFrame(
        {"SMILES": ["a", "b", "c"], "label1": [0, 1, 1], "label2": [1, 1, 0]}
    )
    smiles_list, y = get_mol_strings_and_labels(df)
    assert smiles_list == ["a", "b", "c"]
    assert y.ndim == 2
    assert np.array_equal(y, np.array([[0, 1], [1, 1], [1, 0]]))
