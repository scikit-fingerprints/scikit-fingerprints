import numpy as np
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint
from scipy.sparse import csr_array

from skfp.fingerprints import MACCSFingerprint


def test_maccs_bit_fingerprint(smiles_list, mols_list):
    maccs_fp = MACCSFingerprint(sparse=False, n_jobs=-1)
    X_skfp = maccs_fp.transform(smiles_list)
    X_rdkit = np.array([GetMACCSKeysFingerprint(mol) for mol in mols_list])
    assert np.array_equal(X_skfp, X_rdkit)
    assert X_skfp.shape == (len(smiles_list), 167)


def test_maccs_sparse_bit_fingerprint(smiles_list, mols_list):
    maccs_fp = MACCSFingerprint(sparse=True, n_jobs=-1)
    X_skfp = maccs_fp.transform(smiles_list)
    X_rdkit = csr_array([GetMACCSKeysFingerprint(mol) for mol in mols_list])
    assert np.array_equal(X_skfp.data, X_rdkit.data)
    assert X_skfp.shape == (len(smiles_list), 167)
