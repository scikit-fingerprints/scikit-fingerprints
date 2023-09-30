import numpy as np
import pytest
import rdkit.Chem.rdFingerprintGenerator as fpgens
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint
from rdkit.Chem.rdReducedGraphs import GetErGFingerprint
from scipy.sparse import csr_array

from featurizers.fingerprints import (
    AtomPairFingerprint,
    ERGFingerprint,
    MACCSKeysFingerprint,
    MorganFingerprint,
    TopologicalTorsionFingerprint,
)

smiles_data = [
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


@pytest.fixture
def example_molecules():
    return smiles_data


@pytest.fixture
def rdkit_example_molecules():
    return [Chem.MolFromSmiles(x) for x in smiles_data]


def test_morgan_bit_fingerprint(example_molecules, rdkit_example_molecules):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    fp_gen = fpgens.GetMorganGenerator()
    morgan = MorganFingerprint(n_jobs=-1, sparse=False, count=False)
    X_emf = morgan.transform(X)
    X_rdkit = np.array([fp_gen.GetFingerprint(x) for x in X_for_rdkit])
    assert np.all(X_emf == X_rdkit)


# WARNING - in case of failure it will try to overload memory and result in error
def test_morgan_sparse_fingerprint(example_molecules, rdkit_example_molecules):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    fp_gen = fpgens.GetMorganGenerator()
    morgan = MorganFingerprint(n_jobs=-1, sparse=True, count=False)
    X_emf = morgan.transform(X)
    X_rdkit = csr_array([fp_gen.GetFingerprint(x) for x in X_for_rdkit])
    assert np.all(X_emf.toarray() == X_rdkit.toarray())


def test_morgan_count_fingerprint(example_molecules, rdkit_example_molecules):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    fp_gen = fpgens.GetMorganGenerator()
    morgan = MorganFingerprint(n_jobs=-1, sparse=False, count=True)
    X_emf = morgan.transform(X)
    X_rdkit = np.array(
        [fp_gen.GetCountFingerprint(x).ToList() for x in X_for_rdkit]
    )
    assert np.all(X_emf == X_rdkit)


def test_morgan_sparse_count_fingerprint(
    example_molecules, rdkit_example_molecules
):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    fp_gen = fpgens.GetMorganGenerator()
    morgan = MorganFingerprint(n_jobs=-1, sparse=True, count=True)
    X_emf = morgan.transform(X)
    X_rdkit = csr_array(
        [fp_gen.GetCountFingerprint(x).ToList() for x in X_for_rdkit]
    )
    assert np.all(X_emf.toarray() == X_rdkit.toarray())


def test_atom_pair_bit_fingerprint(example_molecules, rdkit_example_molecules):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    fp_gen = fpgens.GetAtomPairGenerator()
    atom_pair = AtomPairFingerprint(n_jobs=-1, sparse=False, count=False)
    X_emf = atom_pair.transform(X)
    X_rdkit = np.array([fp_gen.GetFingerprint(x) for x in X_for_rdkit])
    assert np.all(X_emf == X_rdkit)


def test_atom_pair_sparse_fingerprint(
    example_molecules, rdkit_example_molecules
):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    fp_gen = fpgens.GetAtomPairGenerator()
    atom_pair = AtomPairFingerprint(n_jobs=-1, sparse=True, count=False)
    X_emf = atom_pair.transform(X)
    X_rdkit = csr_array([fp_gen.GetFingerprint(x) for x in X_for_rdkit])
    assert np.all(X_emf.toarray() == X_rdkit.toarray())


def test_atom_pair_cound_fingerprint(
    example_molecules, rdkit_example_molecules
):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    fp_gen = fpgens.GetAtomPairGenerator()
    atom_pair = AtomPairFingerprint(n_jobs=-1, sparse=False, count=True)
    X_emf = atom_pair.transform(X)
    X_rdkit = np.array(
        [fp_gen.GetCountFingerprint(x).ToList() for x in X_for_rdkit]
    )
    assert np.all(X_emf == X_rdkit)


def test_atom_pair_sparse_count_fingerprint(
    example_molecules, rdkit_example_molecules
):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    fp_gen = fpgens.GetAtomPairGenerator()
    atom_pair = AtomPairFingerprint(n_jobs=-1, sparse=True, count=True)
    X_emf = atom_pair.transform(X)
    X_rdkit = csr_array(
        [fp_gen.GetCountFingerprint(x).ToList() for x in X_for_rdkit]
    )
    assert np.all(X_emf.toarray() == X_rdkit.toarray())


def test_topological_torsion_bit_fingerprint(
    example_molecules, rdkit_example_molecules
):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    fp_gen = fpgens.GetTopologicalTorsionGenerator()
    topological_torsion = TopologicalTorsionFingerprint(
        n_jobs=-1, sparse=False, count=False
    )
    X_emf = topological_torsion.transform(X)
    X_rdkit = np.array([fp_gen.GetFingerprint(x) for x in X_for_rdkit])
    assert np.all(X_emf == X_rdkit)


def test_topological_torsion_sparse_fingerprint(
    example_molecules, rdkit_example_molecules
):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    fp_gen = fpgens.GetTopologicalTorsionGenerator()
    topological_torsion = TopologicalTorsionFingerprint(
        n_jobs=-1, sparse=True, count=False
    )
    X_emf = topological_torsion.transform(X)
    X_rdkit = csr_array([fp_gen.GetFingerprint(x) for x in X_for_rdkit])
    assert np.all(X_emf.toarray() == X_rdkit.toarray())


def test_topological_torsion_count_fingerprint(
    example_molecules, rdkit_example_molecules
):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    fp_gen = fpgens.GetTopologicalTorsionGenerator()
    topological_torsion = TopologicalTorsionFingerprint(
        n_jobs=-1, sparse=False, count=True
    )
    X_emf = topological_torsion.transform(X)
    X_rdkit = np.array(
        [fp_gen.GetCountFingerprint(x).ToList() for x in X_for_rdkit]
    )
    assert np.all(X_emf == X_rdkit)


def test_topological_torsion_sparse_count_fingerprint(
    example_molecules, rdkit_example_molecules
):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    fp_gen = fpgens.GetTopologicalTorsionGenerator()
    topological_torsion = TopologicalTorsionFingerprint(
        n_jobs=-1, sparse=True, count=True
    )
    X_emf = topological_torsion.transform(X)
    X_rdkit = csr_array(
        [fp_gen.GetCountFingerprint(x).ToList() for x in X_for_rdkit]
    )
    assert np.all(X_emf.toarray() == X_rdkit.toarray())


def test_maccs_keys_fingerprint(example_molecules, rdkit_example_molecules):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    erg = MACCSKeysFingerprint(n_jobs=-1, sparse=False)
    X_emf = erg.transform(X)
    X_rdkit = np.array([GetMACCSKeysFingerprint(x) for x in X_for_rdkit])

    assert np.all(X_emf == X_rdkit)


def test_maccs_keys_sparse_fingerprint(
    example_molecules, rdkit_example_molecules
):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    erg = MACCSKeysFingerprint(n_jobs=-1, sparse=True)
    X_emf = erg.transform(X)
    X_rdkit = csr_array([GetMACCSKeysFingerprint(x) for x in X_for_rdkit])
    assert np.all(X_emf.toarray() == X_rdkit.toarray())


def test_erg_fingerprint(example_molecules, rdkit_example_molecules):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    erg = ERGFingerprint(n_jobs=-1, sparse=False)
    X_emf = erg.transform(X)
    X_rdkit = np.array([GetErGFingerprint(x) for x in X_for_rdkit])

    assert np.all(X_emf == X_rdkit)


def test_erg_sparse_fingerprint(example_molecules, rdkit_example_molecules):
    X = example_molecules
    X_for_rdkit = rdkit_example_molecules
    erg = ERGFingerprint(n_jobs=-1, sparse=True)
    X_emf = erg.transform(X)
    X_rdkit = csr_array([GetErGFingerprint(x) for x in X_for_rdkit])
    assert np.all(X_emf.toarray() == X_rdkit.toarray())


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
