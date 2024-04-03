import numpy as np
import scipy.sparse

from skfp.fingerprints import E3FPFingerprint


def test_e3fp_fingerprint(smallest_smiles_list):
    e3fp_fp = E3FPFingerprint(sparse=False, n_jobs=-1)
    X_skfp = e3fp_fp.transform(smallest_smiles_list)

    X_e3fp = np.stack(
        [e3fp_fp._calculate_single_mol_fingerprint(smi) for smi in smallest_smiles_list]
    )

    assert np.array_equal(X_skfp, X_e3fp)
    assert X_skfp.shape == (len(smallest_smiles_list), e3fp_fp.fp_size)
    assert X_skfp.dtype == np.uint8
    assert np.all(np.isin(X_skfp, [0, 1]))


def test_e3fp_sparse_fingerprint(smallest_smiles_list):
    e3fp_fp = E3FPFingerprint(sparse=True, n_jobs=-1)
    X_skfp = e3fp_fp.transform(smallest_smiles_list)

    X_e3fp = scipy.sparse.vstack(
        [e3fp_fp._calculate_single_mol_fingerprint(smi) for smi in smallest_smiles_list]
    )

    assert np.array_equal(X_skfp.data, X_e3fp.data)
    assert X_skfp.shape == (len(smallest_smiles_list), e3fp_fp.fp_size)
    assert X_skfp.dtype == np.uint8
    assert np.all(np.isin(X_skfp.data, [0, 1]))
