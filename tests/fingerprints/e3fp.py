import numpy as np
import scipy.sparse

from skfp import E3FPFingerprint


def test_e3fp(smiles_list):
    e3fp_fp = E3FPFingerprint(
        sparse=False,
        verbose=0,
        n_jobs=-1,
    )
    X_skfp = e3fp_fp.transform(smiles_list)

    X_e3fp = np.stack(
        [e3fp_fp._calculate_single_mol_fingerprint(smi) for smi in smiles_list]
    )

    assert np.array_equal(X_skfp, X_e3fp)


def test_e3fp_sparse(smiles_list):
    e3fp_fp = E3FPFingerprint(
        sparse=True,
        verbose=0,
        n_jobs=-1,
    )
    X_skfp = e3fp_fp.transform(smiles_list)

    X_e3fp = scipy.sparse.vstack([e3fp_fp._calculate_single_mol_fingerprint(smi) for smi in smiles_list])

    assert np.array_equal(X_skfp, X_e3fp)
