import numpy as np
from rdkit.Chem.rdFingerprintGenerator import GetAtomPairGenerator
from scipy.sparse import csr_array

from skfp.fingerprints import AtomPairFingerprint


def test_atom_pair_bit_fingerprint(smiles_list, mols_list):
    atom_pair_fp = AtomPairFingerprint(sparse=False, count=False, n_jobs=-1)
    X_skfp = atom_pair_fp.transform(smiles_list)

    fp_gen = GetAtomPairGenerator(countSimulation=False)
    X_rdkit = np.array([fp_gen.GetFingerprintAsNumPy(mol) for mol in mols_list])

    assert np.array_equal(X_skfp, X_rdkit)
    assert X_skfp.shape == (len(smiles_list), atom_pair_fp.fp_size)


def test_atom_pair_count_fingerprint(smiles_list, mols_list):
    atom_pair_fp = AtomPairFingerprint(sparse=False, count=True, n_jobs=-1)
    X_skfp = atom_pair_fp.transform(smiles_list)

    fp_gen = GetAtomPairGenerator(countSimulation=True)
    X_rdkit = np.array([fp_gen.GetCountFingerprintAsNumPy(mol) for mol in mols_list])

    assert np.array_equal(X_skfp, X_rdkit)
    assert X_skfp.shape == (len(smiles_list), atom_pair_fp.fp_size)


def test_atom_pair_sparse_bit_fingerprint(smiles_list, mols_list):
    atom_pair_fp = AtomPairFingerprint(sparse=True, count=False, n_jobs=-1)
    X_skfp = atom_pair_fp.transform(smiles_list)

    fp_gen = GetAtomPairGenerator(countSimulation=False)
    X_rdkit = csr_array([fp_gen.GetFingerprintAsNumPy(mol) for mol in mols_list])

    assert np.array_equal(X_skfp.data, X_rdkit.data)
    assert X_skfp.shape == (len(smiles_list), atom_pair_fp.fp_size)


def test_atom_pair_sparse_count_fingerprint(smiles_list, mols_list):
    atom_pair_fp = AtomPairFingerprint(sparse=True, count=True, n_jobs=-1)
    X_skfp = atom_pair_fp.transform(smiles_list)

    fp_gen = GetAtomPairGenerator(countSimulation=True)
    X_rdkit = csr_array([fp_gen.GetCountFingerprintAsNumPy(mol) for mol in mols_list])

    assert np.array_equal(X_skfp.data, X_rdkit.data)
    assert X_skfp.shape == (len(smiles_list), atom_pair_fp.fp_size)
