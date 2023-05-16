import pytest

import numpy as np
import pandas as pd

from ogb.graphproppred import PygGraphPropPredDataset
from rdkit.Chem import AllChem
from rdkit import Chem

from featurizers.fingerprints import MorganFingerprintAsBitVect


def test_morgan_fingerprint():
    dataset_name = "ogbg-molhiv"
    PygGraphPropPredDataset(name=dataset_name, root="./test_datasets")

    dataset = pd.read_csv(
        f"./test_datasets/{'_'.join(dataset_name.split('-'))}/mapping/mol.csv.gz"
    )
    X = dataset["smiles"]

    # Copy X for second sequential calculation
    X_2 = X.copy()

    # Concurrent
    morgan = MorganFingerprintAsBitVect(radius=2, n_bits=2048, n_jobs=-1)
    X = np.array([Chem.MolFromSmiles(x) for x in X])
    X_train = morgan.transform(X)

    # Sequential
    X_train_2 = [Chem.MolFromSmiles(x) for x in X_2]
    X_train_2 = [
        AllChem.GetMorganFingerprintAsBitVect(x, 2) for x in X_train_2
    ]
    X_train_2 = np.array(X_train_2)

    # Check if the difference is zero
    diff = np.abs(X_train - X_train_2).sum()
    assert diff == 0
