import pytest

import numpy as np

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

from featurizers.fingerprints import (
    MorganFingerprint,
    MACCSKeysFingerprint,
    AtomPairFingerprint,
    TopologicalTorsionFingerprint,
    ERGFingerprint,
)
from rdkit.Chem.rdReducedGraphs import GetErGFingerprint
import rdkit.Chem.rdFingerprintGenerator as fpgens


@pytest.fixture
def example_molecules():
    return [
        "COc1cccc(NC(=O)CC(=O)N2N=C(N(CCC#N)c3ccc(Cl)cc3)CC2c2ccccc2)c1",
        "CCN(CCO)CCNc1ccc(C)c2sc3ccccc3c(=O)c12",
        "Oc1ncnc2c1sc1nc3ccccc3n12",
        "CC1=CC(=C(c2cc(C)c(O)c(C(=O)O)c2)c2c(Cl)ccc(S(=O)(=O)O)c2Cl)C=C(C(=O)O)C1=O.[NaH]",
        "Cc1ccc2nsnc2c1[N+](=O)[O-]",
    ]


def test_morgan_bit_fingerprint(example_molecules):
    X = example_molecules
    X = np.array([Chem.MolFromSmiles(x) for x in X])

    rdkit_generator = fpgens.GetMorganGenerator()
    morgan = MorganFingerprint(result_vector_type="bit", n_jobs=-1)

    fp_function = rdkit_generator.GetFingerprint
    X_emf = morgan.transform(X.copy())
    X_rdkit = np.array([fp_function(x) for x in X])
    assert np.all(X_emf == X_rdkit)

def test_morgan_count_fingerprint(example_molecules):
    X = example_molecules
    X = np.array([Chem.MolFromSmiles(x) for x in X])

    rdkit_generator = fpgens.GetMorganGenerator()
    morgan = MorganFingerprint(result_vector_type="count", n_jobs=-1)

    fp_function = rdkit_generator.GetCountFingerprint
    X_emf = morgan.transform(X.copy())
    X_rdkit = np.array([fp_function(x) for x in X])
    assert np.all(X_emf == X_rdkit)

def test_morgan_sparse_fingerprint(example_molecules):
    X = example_molecules
    X = np.array([Chem.MolFromSmiles(x) for x in X])

    rdkit_generator = fpgens.GetMorganGenerator()
    morgan = MorganFingerprint(result_vector_type="sparse", n_jobs=-1)

    fp_function = rdkit_generator.GetSparseFingerprint
    X_emf = morgan.transform(X.copy())
    X_rdkit = np.array([fp_function(x) for x in X])
    assert np.all(X_emf == X_rdkit)

def test_morgan_sparse_count_fingerprint(example_molecules):
    X = example_molecules
    X = np.array([Chem.MolFromSmiles(x) for x in X])

    rdkit_generator = fpgens.GetMorganGenerator()
    morgan = MorganFingerprint(result_vector_type="sparse_count", n_jobs=-1)

    fp_function = rdkit_generator.GetSparseCountFingerprint
    X_emf = morgan.transform(X.copy())
    X_rdkit = np.array([fp_function(x) for x in X])
    assert np.all(X_emf == X_rdkit)


def test_maccs_keys_fingerprint(example_molecules):
    X = example_molecules

    X_2 = X.copy()

    # Concurrent
    maccs = MACCSKeysFingerprint(n_jobs=-1)
    X = np.array([Chem.MolFromSmiles(x) for x in X])
    X_maccs = maccs.transform(X.copy())

    # Sequential
    X_2 = [Chem.MolFromSmiles(x) for x in X_2]
    X_seq = np.array([GetMACCSKeysFingerprint(x) for x in X_2])

    assert np.all(X_maccs == X_seq)


def test_atom_pair_fingerprint(example_molecules):
    X = example_molecules

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

    assert np.all(X_atom_pair == X_seq)
    assert np.all(X_atom_pair_hashed == X_seq_hashed)
    assert np.all(X_atom_pair_as_bit_vect == X_seq_as_bit_vect)


def test_topological_torsion_fingerprint(example_molecules):
    X = example_molecules

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

    assert np.all(X_topological_torsion == X_seq)
    assert np.all(X_topological_torsion_hashed == X_seq_hashed)
    assert np.all(X_topological_torsion_as_bit_vect == X_seq_as_bit_vect)


def test_erg_fingerprint(example_molecules):
    X = example_molecules

    X_2 = X.copy()

    # Concurrent
    erg = ERGFingerprint(n_jobs=-1)
    X = np.array([Chem.MolFromSmiles(x) for x in X])
    X_erg = erg.transform(X.copy())

    # Sequential
    X_2 = [Chem.MolFromSmiles(x) for x in X_2]
    X_seq = np.array([GetErGFingerprint(x) for x in X_2])

    assert np.all(X_erg == X_seq)


def test_input_validation(example_molecules):
    X = example_molecules

    X_test = X.copy()

    rdkit_generator = fpgens.GetMorganGenerator()
    morgan = MorganFingerprint(n_jobs=-1)
    fp_function = rdkit_generator.GetFingerprint

    X_test = np.array([Chem.MolFromSmiles(x) for x in X_test])
    X_seq = np.array([fp_function(x) for x in X_test])

    # 1) Some of the molecules are still given as SMILES
    X = [Chem.MolFromSmiles(x) if i % 2 == 0 else x for i, x in enumerate(X)]
    X_morgan = morgan.transform(X)
    assert np.all(X_morgan == X_seq)

    # 2) There exists an element in the list, which is not a molecule
    X_2 = np.copy(X)
    X_2[1] = 1

    with pytest.raises(ValueError) as exec_info:
        X_morgan_2 = morgan.transform(X_2)

    assert (
        str(exec_info.value)
        == "Passed value is neither rdkit.Chem.rdChem.Mol nor SMILES"
    )
