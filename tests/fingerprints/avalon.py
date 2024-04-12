import numpy as np
from rdkit.Avalon.pyAvalonTools import GetAvalonCountFP, GetAvalonFP
from scipy.sparse import csr_array

from skfp.fingerprints import AvalonFingerprint


def test_avalon_bit_fingerprint(smiles_list, mols_list):
    avalon_fp = AvalonFingerprint(sparse=False, count=False, n_jobs=-1)
    X_skfp = avalon_fp.transform(smiles_list)
    X_rdkit = np.array([GetAvalonFP(mol) for mol in mols_list])

    assert np.array_equal(X_skfp, X_rdkit)
    assert X_skfp.shape == (len(smiles_list), avalon_fp.fp_size)
    assert X_skfp.dtype == np.uint8
    assert np.all(np.isin(X_skfp, [0, 1]))


def test_avalon_count_fingerprint(smiles_list, mols_list):
    avalon_fp = AvalonFingerprint(sparse=False, count=True, n_jobs=-1)
    X_skfp = avalon_fp.transform(smiles_list)
    X_rdkit = np.array([GetAvalonCountFP(mol).ToList() for mol in mols_list])

    assert np.array_equal(X_skfp, X_rdkit)
    assert X_skfp.shape == (len(smiles_list), avalon_fp.fp_size)
    assert X_skfp.dtype == np.uint32
    assert np.all(X_skfp >= 0)


def test_avalon_sparse_bit_fingerprint(smiles_list, mols_list):
    avalon_fp = AvalonFingerprint(sparse=True, count=False, n_jobs=-1)
    X_skfp = avalon_fp.transform(smiles_list)
    X_rdkit = csr_array([GetAvalonFP(mol) for mol in mols_list])

    assert np.array_equal(X_skfp.data, X_rdkit.data)
    assert X_skfp.shape == (len(smiles_list), avalon_fp.fp_size)
    assert X_skfp.dtype == np.uint8
    assert np.all(X_skfp.data == 1)


def test_avalon_sparse_count_fingerprint(smiles_list, mols_list):
    avalon_fp = AvalonFingerprint(sparse=True, count=True, n_jobs=-1)
    X_skfp = avalon_fp.transform(smiles_list)
    X_rdkit = csr_array([GetAvalonCountFP(mol).ToList() for mol in mols_list])

    assert np.array_equal(X_skfp.data, X_rdkit.data)
    assert X_skfp.shape == (len(smiles_list), avalon_fp.fp_size)
    assert X_skfp.dtype == np.uint32
    assert np.all(X_skfp.data > 0)
