import numpy as np
import scipy.sparse

from skfp.fingerprints import E3FPFingerprint


def test_e3fp_bit_fingerprint(smallest_mols_smiles_list):
    e3fp_fp = E3FPFingerprint(
        sparse=False,
        verbose=0,
        n_jobs=-1,
    )
    X_skfp = e3fp_fp.transform(smallest_mols_smiles_list)

    X_e3fp = np.stack(
        [
            e3fp_fp._calculate_single_mol_fingerprint(smi)
            for smi in smallest_mols_smiles_list
        ]
    )

    assert np.array_equal(X_skfp, X_e3fp)


def test_e3fp_sparse_bit_fingerprint(smallest_mols_smiles_list):
    e3fp_fp = E3FPFingerprint(
        sparse=True,
        verbose=0,
        n_jobs=-1,
    )
    X_skfp = e3fp_fp.transform(smallest_mols_smiles_list)

    X_e3fp = scipy.sparse.vstack(
        [
            e3fp_fp._calculate_single_mol_fingerprint(smi)
            for smi in smallest_mols_smiles_list
        ]
    )

    assert np.array_equal(X_skfp.data, X_e3fp.data)
