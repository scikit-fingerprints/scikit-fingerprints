import pytest

import numpy as np
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint
from rdkit.Chem.rdReducedGraphs import GetErGFingerprint

from rdkit import Chem

from featurizers.fingerprints import (
    MorganFingerprint,
    MACCSKeysFingerprint,
    AtomPairFingerprint,
    TopologicalTorsionFingerprint,
    ERGFingerprint,
)

import rdkit.Chem.rdFingerprintGenerator as fpgens
from scipy.sparse import csr_array


@pytest.fixture
def example_molecules():
    return [
        "Oc1ncnc2c1sc1nc3ccccc3n12",
        "CC1=CC(=C(c2cc(C)c(O)c(C(=O)O)c2)c2c(Cl)ccc(S(=O)(=O)O)c2Cl)C=C(C(=O)O)C1=O.[NaH]",
        "Cc1ccc2nsnc2c1[N+](=O)[O-]",
        "COc1cccc(NC(=O)CC(=O)N2N=C(N(CCC#N)c3ccc(Cl)cc3)CC2c2ccccc2)c1",
        "CCN(CCO)CCNc1ccc(C)c2sc3ccccc3c(=O)c12",
        "Oc1ncnc2c1sc1nc3ccccc3n12",
        "CC1=CC(=C(c2cc(C)c(O)c(C(=O)O)c2)c2c(Cl)ccc(S(=O)(=O)O)c2Cl)C=C(C(=O)O)C1=O.[NaH]",
        "Cc1ccc2nsnc2c1[N+](=O)[O-]",
        "COc1cccc(NC(=O)CC(=O)N2N=C(N(CCC#N)c3ccc(Cl)cc3)CC2c2ccccc2)c1",
        "CCN(CCO)CCNc1ccc(C)c2sc3ccccc3c(=O)c12",
        "Oc1ncnc2c1sc1nc3ccccc3n12",
        "CC1=CC(=C(c2cc(C)c(O)c(C(=O)O)c2)c2c(Cl)ccc(S(=O)(=O)O)c2Cl)C=C(C(=O)O)C1=O.[NaH]",
        "Cc1ccc2nsnc2c1[N+](=O)[O-]",
        "COc1cccc(NC(=O)CC(=O)N2N=C(N(CCC#N)c3ccc(Cl)cc3)CC2c2ccccc2)c1",
        "CCN(CCO)CCNc1ccc(C)c2sc3ccccc3c(=O)c12",
        "Oc1ncnc2c1sc1nc3ccccc3n12",
        "CC1=CC(=C(c2cc(C)c(O)c(C(=O)O)c2)c2c(Cl)ccc(S(=O)(=O)O)c2Cl)C=C(C(=O)O)C1=O.[NaH]",
    ]


def test_morgan_bit_fingerprint(example_molecules):
    X = example_molecules
    X_for_rdkit = [Chem.MolFromSmiles(x) for x in X]
    rdkit_generator = fpgens.GetMorganGenerator()
    morgan = MorganFingerprint(fingerprint_type="bit", n_jobs=-1)
    fp_function = rdkit_generator.GetFingerprint
    X_emf = morgan.transform(X.copy())
    X_rdkit = np.array([fp_function(x) for x in X_for_rdkit])
    assert np.all(X_emf == X_rdkit)


# WARNING - in case of failure it will try to overload memory and result in error
def test_morgan_sparse_fingerprint(example_molecules):
    X = example_molecules
    X_for_rdkit = [Chem.MolFromSmiles(x) for x in X]
    rdkit_generator = fpgens.GetMorganGenerator()
    morgan = MorganFingerprint(fingerprint_type="bit", sparse=True, n_jobs=-1)
    fp_function = rdkit_generator.GetFingerprint
    X_emf = morgan.transform(X.copy())
    X_rdkit = csr_array([fp_function(x) for x in X_for_rdkit])
    assert np.all(X_emf.toarray() == X_rdkit.toarray())


def test_morgan_count_fingerprint(example_molecules):
    X = example_molecules
    X_for_rdkit = [Chem.MolFromSmiles(x) for x in X]
    rdkit_generator = fpgens.GetMorganGenerator()
    morgan = MorganFingerprint(fingerprint_type="count", n_jobs=-1)
    fp_function = rdkit_generator.GetCountFingerprint
    X_emf = morgan.transform(X.copy())
    X_rdkit = np.array([fp_function(x).ToList() for x in X_for_rdkit])
    assert np.all(X_emf == X_rdkit)


def test_morgan_sparse_count_fingerprint(example_molecules):
    X = example_molecules
    X_for_rdkit = [Chem.MolFromSmiles(x) for x in X]
    rdkit_generator = fpgens.GetMorganGenerator()
    morgan = MorganFingerprint(
        fingerprint_type="count", sparse=True, n_jobs=-1
    )
    fp_function = rdkit_generator.GetCountFingerprint
    X_emf = morgan.transform(X.copy())
    X_rdkit = csr_array([fp_function(x).ToList() for x in X_for_rdkit])
    assert np.all(X_emf.toarray() == X_rdkit.toarray())


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


def test_atom_pair_bit_fingerprint(example_molecules):
    X = example_molecules
    X_for_rdkit = [Chem.MolFromSmiles(x) for x in X]
    rdkit_generator = fpgens.GetAtomPairGenerator()
    atom_pair = AtomPairFingerprint(fingerprint_type="bit", n_jobs=-1)
    fp_function = rdkit_generator.GetFingerprint
    X_emf = atom_pair.transform(X.copy())
    X_rdkit = np.array([fp_function(x) for x in X_for_rdkit])
    assert np.all(X_emf == X_rdkit)


def test_atom_pair_sparse_fingerprint(example_molecules):
    X = example_molecules
    X_for_rdkit = [Chem.MolFromSmiles(x) for x in X]
    rdkit_generator = fpgens.GetAtomPairGenerator()
    atom_pair = AtomPairFingerprint(
        fingerprint_type="bit", sparse=True, n_jobs=-1
    )
    fp_function = rdkit_generator.GetFingerprint
    X_emf = atom_pair.transform(X.copy())
    X_rdkit = csr_array([fp_function(x) for x in X_for_rdkit])
    assert np.all(X_emf.toarray() == X_rdkit.toarray())


def test_atom_pair_cound_fingerprint(example_molecules):
    X = example_molecules
    X_for_rdkit = [Chem.MolFromSmiles(x) for x in X]
    rdkit_generator = fpgens.GetAtomPairGenerator()
    atom_pair = AtomPairFingerprint(fingerprint_type="count", n_jobs=-1)
    fp_function = rdkit_generator.GetCountFingerprint
    X_emf = atom_pair.transform(X.copy())
    X_rdkit = np.array([fp_function(x).ToList() for x in X_for_rdkit])
    assert np.all(X_emf == X_rdkit)


def test_atom_pair_sparse_count_fingerprint(example_molecules):
    X = example_molecules
    X_for_rdkit = [Chem.MolFromSmiles(x) for x in X]
    rdkit_generator = fpgens.GetAtomPairGenerator()
    atom_pair = AtomPairFingerprint(
        fingerprint_type="count", sparse=True, n_jobs=-1
    )
    fp_function = rdkit_generator.GetCountFingerprint
    X_emf = atom_pair.transform(X.copy())
    X_rdkit = csr_array([fp_function(x).ToList() for x in X_for_rdkit])
    assert np.all(X_emf.toarray() == X_rdkit.toarray())


def test_topological_torsion_bit_fingerprint(example_molecules):
    X = example_molecules
    X_for_rdkit = [Chem.MolFromSmiles(x) for x in X]
    rdkit_generator = fpgens.GetTopologicalTorsionGenerator()
    topological_torsion = TopologicalTorsionFingerprint(
        fingerprint_type="bit", n_jobs=-1
    )
    fp_function = rdkit_generator.GetFingerprint
    X_emf = topological_torsion.transform(X.copy())
    X_rdkit = np.array([fp_function(x) for x in X_for_rdkit])
    assert np.all(X_emf == X_rdkit)


def test_topological_torsion_sparse_fingerprint(example_molecules):
    X = example_molecules
    X_for_rdkit = [Chem.MolFromSmiles(x) for x in X]
    rdkit_generator = fpgens.GetTopologicalTorsionGenerator()
    topological_torsion = TopologicalTorsionFingerprint(
        fingerprint_type="bit", sparse=True, n_jobs=-1
    )
    fp_function = rdkit_generator.GetFingerprint
    X_emf = topological_torsion.transform(X.copy())
    X_rdkit = csr_array([fp_function(x) for x in X_for_rdkit])
    assert np.all(X_emf.toarray() == X_rdkit.toarray())


def test_topological_torsion_count_fingerprint(example_molecules):
    X = example_molecules
    X_for_rdkit = [Chem.MolFromSmiles(x) for x in X]
    rdkit_generator = fpgens.GetTopologicalTorsionGenerator()
    topological_torsion = TopologicalTorsionFingerprint(
        fingerprint_type="count", n_jobs=-1
    )
    fp_function = rdkit_generator.GetCountFingerprint
    X_emf = topological_torsion.transform(X.copy())
    X_rdkit = np.array([fp_function(x).ToList() for x in X_for_rdkit])
    assert np.all(X_emf == X_rdkit)


def test_topological_torsion_sparse_count_fingerprint(example_molecules):
    X = example_molecules
    X_for_rdkit = [Chem.MolFromSmiles(x) for x in X]
    rdkit_generator = fpgens.GetTopologicalTorsionGenerator()
    topological_torsion = TopologicalTorsionFingerprint(
        fingerprint_type="count", sparse=True, n_jobs=-1
    )
    fp_function = rdkit_generator.GetCountFingerprint
    X_emf = topological_torsion.transform(X.copy())
    X_rdkit = csr_array([fp_function(x).ToList() for x in X_for_rdkit])
    assert np.all(X_emf.toarray() == X_rdkit.toarray())


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
