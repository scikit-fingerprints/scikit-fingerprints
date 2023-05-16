import pytest

import numpy as np
import pandas as pd

from ogb.graphproppred import PygGraphPropPredDataset
from rdkit.Chem.rdMolDescriptors import (
    GetMorganFingerprint,
    GetMorganFingerprintAsBitVect,
    GetHashedMorganFingerprint,
    GetMACCSKeysFingerprint,
    GetAtomPairFingerprint,
    GetHashedAtomPairFingerprint,
    GetHashedAtomPairFingerprintAsBitVect,
    GetTopologicalTorsionFingerprint,
    GetHashedTopologicalTorsionFingerprint,
    GetHashedTopologicalTorsionFingerprintAsBitVect,
)
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw

from featurizers.fingerprints import (
    MorganFingerprint,
    MACCSKeysFingerprint,
    AtomPairFingerprint,
    TopologicalTorsionFingerprint,
    ERGFingerprint,
)
from rdkit.Chem.rdReducedGraphs import GetErGFingerprint


def dice_similarity_matrix(A, B):
    similarity_sum = 0
    for i in range(len(A)):
        next_sum = DataStructs.DiceSimilarity(A[i], B[i])
        similarity_sum += next_sum

        if next_sum == 0.0:
            print(i)
            print(A[i])
            print(B[i])
            print(DataStructs.TanimotoSimilarity(A[i], B[i]))

    return similarity_sum / len(A)


def absolute_error(A, B):
    return np.sum(np.abs(A - B))


def test_morgan_fingerprint():
    dataset_name = "ogbg-molhiv"
    PygGraphPropPredDataset(name=dataset_name, root="./test_datasets")

    dataset = pd.read_csv(
        f"./test_datasets/{'_'.join(dataset_name.split('-'))}/mapping/mol.csv.gz"
    )
    X = dataset["smiles"]
    X = X[:100]

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
    X_seq = np.array([GetMorganFingerprint(x, 2) for x in X_2])
    X_seq_hashed = np.array([GetHashedMorganFingerprint(x, 2) for x in X_2])
    X_seq_as_bit_vect = np.array(
        [GetMorganFingerprintAsBitVect(x, 2) for x in X_2]
    )

    assert dice_similarity_matrix(X_morgan, X_seq) == 1.0
    assert dice_similarity_matrix(X_morgan_hashed, X_seq_hashed) == 1.0
    assert absolute_error(X_morgan_as_bit_vect, X_seq_as_bit_vect) == 0


def test_maccs_keys_fingerprint():
    dataset_name = "ogbg-molhiv"
    PygGraphPropPredDataset(name=dataset_name, root="./test_datasets")

    dataset = pd.read_csv(
        f"./test_datasets/{'_'.join(dataset_name.split('-'))}/mapping/mol.csv.gz"
    )
    X = dataset["smiles"]
    X = X[:100]

    X_2 = X.copy()

    # Concurrent
    maccs = MACCSKeysFingerprint(n_jobs=-1)
    X = np.array([Chem.MolFromSmiles(x) for x in X])
    X_maccs = maccs.transform(X.copy())

    # Sequential
    X_2 = [Chem.MolFromSmiles(x) for x in X_2]
    X_seq = np.array([GetMACCSKeysFingerprint(x) for x in X_2])

    assert absolute_error(X_maccs, X_seq) == 0


def test_atom_pair_fingerprint():
    dataset_name = "ogbg-molhiv"
    PygGraphPropPredDataset(name=dataset_name, root="./test_datasets")

    dataset = pd.read_csv(
        f"./test_datasets/{'_'.join(dataset_name.split('-'))}/mapping/mol.csv.gz"
    )
    X = dataset["smiles"]
    X = X[:100]

    X_2 = X.copy()

    # Concurrent
    atom_pair = AtomPairFingerprint(result_type="default", n_jobs=-1)
    atom_pair_hashed = AtomPairFingerprint(result_type="hashed", n_jobs=-1)
    atom_pair_as_bit_vect = AtomPairFingerprint(
        result_type="as_bit_vect", n_jobs=-1
    )
    X = np.array([Chem.MolFromSmiles(x) for x in X])
    X_atom_pair = atom_pair.transform(X.copy())
    X_atom_pair_hashed = atom_pair_hashed.transform(X.copy())
    X_atom_pair_as_bit_vect = atom_pair_as_bit_vect.transform(X.copy())

    # Sequential
    X_2 = [Chem.MolFromSmiles(x) for x in X_2]
    X_seq = np.array([GetAtomPairFingerprint(x) for x in X_2])
    X_seq_hashed = np.array([GetHashedAtomPairFingerprint(x) for x in X_2])
    X_seq_as_bit_vect = np.array(
        [GetHashedAtomPairFingerprintAsBitVect(x) for x in X_2]
    )

    assert dice_similarity_matrix(X_atom_pair, X_seq) == 1.0
    assert dice_similarity_matrix(X_atom_pair_hashed, X_seq_hashed) == 1.0
    assert absolute_error(X_atom_pair_as_bit_vect, X_seq_as_bit_vect) == 0


def test_topological_torsion_fingerprint():
    dataset_name = "ogbg-molhiv"
    PygGraphPropPredDataset(name=dataset_name, root="./test_datasets")

    dataset = pd.read_csv(
        f"./test_datasets/{'_'.join(dataset_name.split('-'))}/mapping/mol.csv.gz"
    )
    X = dataset["smiles"]
    X = X[:100]

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
    X_seq = np.array([GetTopologicalTorsionFingerprint(x) for x in X_2])
    X_seq_hashed = np.array(
        [GetHashedTopologicalTorsionFingerprint(x) for x in X_2]
    )
    X_seq_as_bit_vect = np.array(
        [GetHashedTopologicalTorsionFingerprintAsBitVect(x) for x in X_2]
    )

    # TODO - fingerprint for molecule at position 40 in X is calculated, but dice similarity is 0.0
    # assert dice_similarity_matrix(X_topological_torsion, X_seq) == 1.0
    # assert dice_similarity_matrix(X_topological_torsion_hashed, X_seq_hashed) == 1.0
    assert (
        absolute_error(X_topological_torsion_as_bit_vect, X_seq_as_bit_vect)
        == 0
    )


def test_erg_fingerprint():
    dataset_name = "ogbg-molhiv"
    PygGraphPropPredDataset(name=dataset_name, root="./test_datasets")

    dataset = pd.read_csv(
        f"./test_datasets/{'_'.join(dataset_name.split('-'))}/mapping/mol.csv.gz"
    )
    X = dataset["smiles"]
    X = X[:100]

    X_2 = X.copy()

    # Concurrent
    erg = ERGFingerprint(n_jobs=-1)
    X = np.array([Chem.MolFromSmiles(x) for x in X])
    X_erg = erg.transform(X.copy())

    # Sequential
    X_2 = [Chem.MolFromSmiles(x) for x in X_2]
    X_seq = np.array([GetErGFingerprint(x) for x in X_2])

    assert absolute_error(X_erg, X_seq) == 0
