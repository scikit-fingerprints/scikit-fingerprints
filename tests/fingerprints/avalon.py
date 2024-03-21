import numpy as np
from rdkit.Avalon.pyAvalonTools import GetAvalonCountFP, GetAvalonFP
from scipy.sparse import csr_array

from skfp.fingerprints import AvalonFingerprint


def test_avalon_bit_fingerprint(smiles_list, mols_list):
    avalon_fp = AvalonFingerprint(sparse=False, count=False, n_jobs=-1)
    X_skfp = avalon_fp.transform(smiles_list)
    X_rdkit = np.array([GetAvalonFP(mol) for mol in mols_list])
    assert np.array_equal(X_skfp, X_rdkit)


def test_avalon_count_fingerprint(smiles_list, mols_list):
    avalon_fp = AvalonFingerprint(sparse=False, count=True, n_jobs=-1)
    X_skfp = avalon_fp.transform(smiles_list)
    X_rdkit = np.array([GetAvalonCountFP(mol).ToList() for mol in mols_list])
    assert np.array_equal(X_skfp, X_rdkit)


def test_avalon_sparse_bit_fingerprint(smiles_list, mols_list):
    avalon_fp = AvalonFingerprint(sparse=True, count=False, n_jobs=-1)
    X_skfp = avalon_fp.transform(smiles_list)
    X_rdkit = csr_array([GetAvalonFP(mol) for mol in mols_list])
    assert np.array_equal(X_skfp.data, X_rdkit.data)


def test_avalon_sparse_count_fingerprint(smiles_list, mols_list):
    avalon_fp = AvalonFingerprint(sparse=True, count=True, n_jobs=-1)
    X_skfp = avalon_fp.transform(smiles_list)
    X_rdkit = csr_array([GetAvalonCountFP(mol).ToList() for mol in mols_list])
    assert np.array_equal(X_skfp.data, X_rdkit.data)
