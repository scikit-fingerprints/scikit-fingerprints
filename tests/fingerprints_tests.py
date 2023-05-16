import pytest

import numpy as np
import pandas as pd

from ogb.graphproppred import PygGraphPropPredDataset
from rdkit.Chem import AllChem
from rdkit import Chem

from featurizers.fingerprints import (
    MorganFingerprint,
    MACCSKeysFingerprint,
    AtomPairFingerprint,
    TopologicalTorsionFingerprint,
    ERGFingerprint,
)
from rdkit.Chem.rdReducedGraphs import GetErGFingerprint


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
    morgan = MorganFingerprint(result_type="default", n_jobs=-1)
    morgan_hashed = MorganFingerprint(result_type="hashed", n_jobs=-1)
    morgan_as_bit_vect = MorganFingerprint(
        result_type="as_bit_vect", n_jobs=-1
    )
    X = np.array([Chem.MolFromSmiles(x) for x in X])
    X_morgan = morgan.transform(X.copy())
    X_morgan_hashed = morgan_hashed.transform(X.copy())
    X_morgan_as_bit_vect = morgan_as_bit_vect.transform(X.copy())

    # Sequential
    X_2 = [Chem.MolFromSmiles(x) for x in X_2]
    X_seq = np.array([AllChem.GetMorganFingerprint(x, 2) for x in X_2])
    X_seq_hashed = np.array(
        [AllChem.GetHashedMorganFingerprint(x, 2) for x in X_2]
    )
    X_seq_as_bit_vect = np.array(
        [AllChem.GetMorganFingerprintAsBitVect(x, 2) for x in X_2]
    )

    # TODO - Check if the difference is zero


def test_maccs_keys_fingerprint():
    dataset_name = "ogbg-molhiv"
    PygGraphPropPredDataset(name=dataset_name, root="./test_datasets")

    dataset = pd.read_csv(
        f"./test_datasets/{'_'.join(dataset_name.split('-'))}/mapping/mol.csv.gz"
    )
    X = dataset["smiles"]

    X_2 = X.copy()

    # Concurrent
    maccs = MACCSKeysFingerprint(n_jobs=-1)
    X = np.array([Chem.MolFromSmiles(x) for x in X])
    X_maccs = maccs.transform(X.copy())

    # Sequential
    X_2 = [Chem.MolFromSmiles(x) for x in X_2]
    X_seq = np.array([AllChem.GetMACCSKeysFingerprint(x) for x in X_2])

    # TODO - Check if the difference is zero


def test_atom_pair_fingerprint():
    dataset_name = "ogbg-molhiv"
    PygGraphPropPredDataset(name=dataset_name, root="./test_datasets")

    dataset = pd.read_csv(
        f"./test_datasets/{'_'.join(dataset_name.split('-'))}/mapping/mol.csv.gz"
    )
    X = dataset["smiles"]

    # Copy X for second sequential calculation
    X_2 = X.copy()

    # Concurrent
    atom_pair = AtomPairFingerprint(result_type="default", n_jobs=-1)
    atom_pair_hashed = MorganFingerprint(result_type="hashed", n_jobs=-1)
    atom_pair_as_bit_vect = MorganFingerprint(
        result_type="as_bit_vect", n_jobs=-1
    )
    X = np.array([Chem.MolFromSmiles(x) for x in X])
    X_atom_pair = atom_pair.transform(X.copy())
    X_atom_pair_hashed = atom_pair_hashed.transform(X.copy())
    X_atom_pair_as_bit_vect = atom_pair_as_bit_vect.transform(X.copy())

    # Sequential
    X_2 = [Chem.MolFromSmiles(x) for x in X_2]
    X_seq = np.array([AllChem.GetAtomPairFingerprint(x) for x in X_2])
    X_seq_hashed = np.array(
        [AllChem.GetHashedAtomPairFingerprint(x) for x in X_2]
    )
    X_seq_as_bit_vect = np.array(
        [AllChem.GetHashedAtomPairFingerprintAsBitVect(x) for x in X_2]
    )

    # TODO - Check if the difference is zero


def test_topological_torsion_fingerprint():
    dataset_name = "ogbg-molhiv"
    PygGraphPropPredDataset(name=dataset_name, root="./test_datasets")

    dataset = pd.read_csv(
        f"./test_datasets/{'_'.join(dataset_name.split('-'))}/mapping/mol.csv.gz"
    )
    X = dataset["smiles"]

    X_2 = X.copy()

    # Concurrent
    topological_torsion = TopologicalTorsionFingerprint(
        result_type="default", n_jobs=-1
    )
    topological_torsion_hashed = TopologicalTorsionFingerprint(
        result_type="hashed", n_jobs=-1
    )
    topological_torsion_as_bit_vect = TopologicalTorsionFingerprint(
        result_type="as_bit_vect", n_jobs=-1
    )
    X = np.array([Chem.MolFromSmiles(x) for x in X])
    X_topological_torsion = topological_torsion.transform(X.copy())
    X_topological_torsion_hashed = topological_torsion_hashed.transform(
        X.copy()
    )
    X_topological_torsion_as_bit_vect = (
        topological_torsion_as_bit_vect.transform(X.copy())
    )

    # Sequential
    X_2 = [Chem.MolFromSmiles(x) for x in X_2]
    X_seq = np.array(
        [AllChem.GetTopologicalTorsionFingerprint(x) for x in X_2]
    )
    X_seq_hashed = np.array(
        [AllChem.GetHashedTopologicalTorsionFingerprint(x) for x in X_2]
    )
    X_seq_as_bit_vect = np.array(
        [
            AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(x)
            for x in X_2
        ]
    )

    # TODO - Check if the difference is zero


def test_erg_fingerprint():
    dataset_name = "ogbg-molhiv"
    PygGraphPropPredDataset(name=dataset_name, root="./test_datasets")

    dataset = pd.read_csv(
        f"./test_datasets/{'_'.join(dataset_name.split('-'))}/mapping/mol.csv.gz"
    )
    X = dataset["smiles"]
    X_2 = X.copy()

    # Concurrent
    erg = ERGFingerprint(n_jobs=-1)
    X = np.array([Chem.MolFromSmiles(x) for x in X])
    X_maccs = erg.transform(X.copy())

    # Sequential
    X_2 = [Chem.MolFromSmiles(x) for x in X_2]
    X_seq = np.array([GetErGFingerprint(x) for x in X_2])

    # TODO - Check if the difference is zero
