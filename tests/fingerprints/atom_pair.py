import numpy as np
from rdkit.Chem.rdFingerprintGenerator import GetAtomPairGenerator
from scipy.sparse import csr_array

from skfp import AtomPairFingerprint
from helpers import sparse_equal


def test_atom_pair_bit_fingerprint(smiles_list, mols_list):
    atom_pair_fp = AtomPairFingerprint(sparse=False, count=False, n_jobs=-1)
    X_skfp = atom_pair_fp.transform(smiles_list)

    fp_gen = GetAtomPairGenerator()
    X_rdkit = np.array([fp_gen.GetFingerprint(mol) for mol in mols_list])

    assert np.array_equal(X_skfp, X_rdkit)


def test_atom_pair_count_fingerprint(smiles_list, mols_list):
    atom_pair_fp = AtomPairFingerprint(sparse=False, count=True, n_jobs=-1)
    X_skfp = atom_pair_fp.transform(smiles_list)

    fp_gen = GetAtomPairGenerator()
    X_rdkit = np.array(
        [fp_gen.GetCountFingerprint(mol).ToList() for mol in mols_list]
    )

    assert np.array_equal(X_skfp, X_rdkit)


def test_atom_pair_sparse_bit_fingerprint(smiles_list, mols_list):
    atom_pair_fp = AtomPairFingerprint(sparse=True, count=False, n_jobs=-1)
    X_skfp = atom_pair_fp.transform(smiles_list)

    fp_gen = GetAtomPairGenerator()
    X_rdkit = csr_array([fp_gen.GetFingerprint(mol) for mol in mols_list])

    assert sparse_equal(X_skfp, X_rdkit)


def test_atom_pair_sparse_count_fingerprint(smiles_list, mols_list):
    atom_pair_fp = AtomPairFingerprint(sparse=True, count=True, n_jobs=-1)
    X_skfp = atom_pair_fp.transform(smiles_list)

    fp_gen = GetAtomPairGenerator()
    X_rdkit = csr_array([fp_gen.GetFingerprint(mol) for mol in mols_list])

    assert sparse_equal(X_skfp, X_rdkit)
