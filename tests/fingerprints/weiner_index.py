import numpy as np
from rdkit.Chem import GetDistanceMatrix
from scipy.sparse import csr_array

from skfp.fingerprints import WeinerIndexFingerprint


def test_weiner_bit_fingerprint(smiles_list, mols_list):
    weiner_fp = WeinerIndexFingerprint(n_jobs=-1)
    X_skfp = weiner_fp.transform(smiles_list)

    X_manual = np.array(
        [np.sum(GetDistanceMatrix(mol)) / 2 if mol else 0.0 for mol in mols_list]
    ).reshape(-1, 1)

    assert np.allclose(X_skfp, X_manual)
    assert X_skfp.shape == (len(smiles_list), 1)
    assert X_skfp.dtype == np.float64


def test_weiner_sparse_bit_fingerprint(smiles_list, mols_list):
    weiner_fp = WeinerIndexFingerprint(sparse=True, n_jobs=-1)
    X_skfp = weiner_fp.transform(smiles_list)

    X_manual_sparse = csr_array(
        [[np.sum(GetDistanceMatrix(mol)) / 2 if mol else 0.0] for mol in mols_list]
    )

    assert np.array_equal(X_skfp.data, X_manual_sparse.data)
    assert X_skfp.shape == (len(smiles_list), 1)
    assert X_skfp.dtype == np.float64
