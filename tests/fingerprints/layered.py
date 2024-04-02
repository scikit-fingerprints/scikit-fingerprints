import numpy as np
from rdkit.Chem.rdmolops import LayeredFingerprint as RDKitLayeredFingerprint
from scipy.sparse import csr_array

from skfp.fingerprints import LayeredFingerprint


def test_layered_fingerprint(smiles_list, mols_list):
    layered_fp = LayeredFingerprint(sparse=False, n_jobs=-1)
    X_skfp = layered_fp.transform(smiles_list)
    X_rdkit = np.array([RDKitLayeredFingerprint(mol) for mol in mols_list])
    assert np.array_equal(X_skfp, X_rdkit)
    assert X_skfp.shape == (len(smiles_list), layered_fp.fp_size)


def test_layered_sparse_fingerprint(smiles_list, mols_list):
    layered_fp = LayeredFingerprint(sparse=True, n_jobs=-1)
    X_skfp = layered_fp.transform(smiles_list)
    X_rdkit = csr_array([RDKitLayeredFingerprint(mol) for mol in mols_list])
    assert np.array_equal(X_skfp.data, X_rdkit.data)
    assert X_skfp.shape == (len(smiles_list), layered_fp.fp_size)
