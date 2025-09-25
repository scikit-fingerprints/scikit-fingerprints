import os
import shutil

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal

from skfp.datasets.utils import (
    fetch_splits,
    get_data_home_dir,
    get_mol_strings_and_labels,
)


def test_get_data_home_dir():
    default_dir = get_data_home_dir(None, dataset_name="test")
    assert default_dir.endswith(os.path.join("scikit_learn_data", "test"))

    try:
        custom_dir = get_data_home_dir(data_dir="./data", dataset_name="test")
    finally:
        shutil.rmtree("data")

    assert_equal(custom_dir, os.path.join("data", "test"))


def test_fetch_splits(capsys):
    fetch_splits(
        None,
        dataset_name="MoleculeNet_BACE",
        filename="ogb_splits_bace.json",
        verbose=True,
    )
    stdout = capsys.readouterr().out
    assert "MoleculeNet_BACE" in stdout
    assert "ogb_splits_bace.json" in stdout


@pytest.mark.parametrize("mol_type", ["SMILES", "aminoseq"])
def test_get_smiles_and_labels(mol_type):
    df = pd.DataFrame({mol_type: ["a", "b", "c"], "label": [0, 0, 1]})
    smiles_list, y = get_mol_strings_and_labels(df, mol_type=mol_type)
    assert_equal(smiles_list, ["a", "b", "c"])
    assert_equal(y.ndim, 1)
    assert_equal(y, np.array([0, 0, 1]))

    df = pd.DataFrame(
        {mol_type: ["a", "b", "c"], "label1": [0, 1, 1], "label2": [1, 1, 0]}
    )
    smiles_list, y = get_mol_strings_and_labels(df, mol_type=mol_type)

    assert_equal(smiles_list, ["a", "b", "c"])
    assert_equal(y.ndim, 2)
    assert_equal(y, np.array([[0, 1], [1, 1], [1, 0]]))


def test_wrong_mol_type():
    with pytest.raises(ValueError, match="mol_type nonexistent not recognized"):
        df = pd.DataFrame({"SMILES": ["a", "b", "c"], "label": [0, 0, 1]})
        get_mol_strings_and_labels(df, mol_type="nonexistent")
