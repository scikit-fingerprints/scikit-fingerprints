import numpy as np
from rdkit.Chem.rdMolDescriptors import CalcWHIM
from scipy.sparse import csr_array

from skfp.fingerprints import WHIMFingerprint


def test_whim_fingerprint(mols_conformers_list):
    whim_fp = WHIMFingerprint(sparse=False, n_jobs=-1)
    X_skfp = whim_fp.transform(mols_conformers_list)

    X_rdkit = np.array(
        [CalcWHIM(mol, confId=mol.conf_id) for mol in mols_conformers_list]
    )
    X_rdkit = np.minimum(X_rdkit, whim_fp.clip_val)

    assert np.allclose(X_skfp, X_rdkit, atol=1e-1)
    assert X_skfp.shape == (len(mols_conformers_list), 114)


def test_whim_sparse_fingerprint(mols_conformers_list):
    whim_fp = WHIMFingerprint(sparse=True, n_jobs=-1)
    X_skfp = whim_fp.transform(mols_conformers_list)

    X_rdkit = csr_array(
        [CalcWHIM(mol, confId=mol.conf_id) for mol in mols_conformers_list]
    )
    X_rdkit = X_rdkit.minimum(whim_fp.clip_val)

    assert np.allclose(X_skfp.data, X_rdkit.data, atol=1e-1)
    assert X_skfp.shape == (len(mols_conformers_list), 114)
