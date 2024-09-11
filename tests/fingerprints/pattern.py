import numpy as np
from rdkit.Chem.rdmolops import PatternFingerprint as RDKitPatternFingerprint
from scipy.sparse import csr_array

from skfp.fingerprints import PatternFingerprint


def test_pattern_fingerprint(smiles_list, mols_list):
    pattern_fp = PatternFingerprint(n_jobs=-1)
    X_skfp = pattern_fp.transform(smiles_list)
    X_rdkit = np.array([RDKitPatternFingerprint(mol) for mol in mols_list])

    assert np.array_equal(X_skfp, X_rdkit)
    assert X_skfp.shape == (len(smiles_list), pattern_fp.fp_size)
    assert X_skfp.dtype == np.uint8
    assert np.all(np.isin(X_skfp, [0, 1]))


def test_pattern_sparse_fingerprint(smiles_list, mols_list):
    pattern_fp = PatternFingerprint(sparse=True, n_jobs=-1)
    X_skfp = pattern_fp.transform(smiles_list)
    X_rdkit = csr_array([RDKitPatternFingerprint(mol) for mol in mols_list])

    assert np.array_equal(X_skfp.data, X_rdkit.data)
    assert X_skfp.shape == (len(smiles_list), pattern_fp.fp_size)
    assert X_skfp.dtype == np.uint8
    assert np.all(X_skfp.data == 1)
