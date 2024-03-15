import numpy as np
from rdkit.Chem.rdMHFPFingerprint import MHFPEncoder
from scipy.sparse import csr_array

from skfp.fingerprints import MHFPFingerprint


def test_mhfp_bit_fingerprint(smiles_list, mols_list):
    mhfp_fp = MHFPFingerprint(sparse=False, n_jobs=-1)
    X_skfp = mhfp_fp.transform(smiles_list)

    encoder = MHFPEncoder(2048, 0)
    X_rdkit = MHFPEncoder.EncodeMolsBulk(
        encoder,
        mols_list,
    )
    X_rdkit = np.array(X_rdkit)

    assert np.array_equal(X_skfp, X_rdkit)


def test_mhfp_sparse_bit_fingerprint(smiles_list, mols_list):
    mhfp_fp = MHFPFingerprint(sparse=True, n_jobs=-1)
    X_skfp = mhfp_fp.transform(smiles_list)

    encoder = MHFPEncoder(2048, 0)
    X_rdkit = MHFPEncoder.EncodeMolsBulk(
        encoder,
        mols_list,
    )
    X_rdkit = csr_array(X_rdkit)

    assert np.array_equal(X_skfp.data, X_rdkit.data)
