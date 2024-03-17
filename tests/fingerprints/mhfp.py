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
    X_rdkit = np.mod(X_rdkit, mhfp_fp.fp_size)
    X_rdkit = np.stack([np.bincount(x, minlength=mhfp_fp.fp_size) for x in X_rdkit])
    X_rdkit = (X_rdkit > 0).astype(int)

    assert np.array_equal(X_skfp, X_rdkit)


def test_mhfp_sparse_bit_fingerprint(smiles_list, mols_list):
    mhfp_fp = MHFPFingerprint(sparse=True, n_jobs=-1)
    X_skfp = mhfp_fp.transform(smiles_list)

    encoder = MHFPEncoder(2048, 0)
    X_rdkit = MHFPEncoder.EncodeMolsBulk(
        encoder,
        mols_list,
    )
    X_rdkit = np.array(X_rdkit)
    X_rdkit = np.mod(X_rdkit, mhfp_fp.fp_size)
    X_rdkit = csr_array(
        [(np.bincount(x, minlength=mhfp_fp.fp_size) > 0) for x in X_rdkit],
        dtype=int,
    )

    assert np.array_equal(X_skfp.data, X_rdkit.data)


def test_mhfp_raw_hashes_fingerprint(smiles_list, mols_list):
    mhfp_fp = MHFPFingerprint(output_raw_hashes=True, sparse=False, n_jobs=-1)
    X_skfp = mhfp_fp.transform(smiles_list)

    encoder = MHFPEncoder(2048, 0)
    X_rdkit = MHFPEncoder.EncodeMolsBulk(
        encoder,
        mols_list,
    )
    X_rdkit = np.array(X_rdkit)

    assert np.array_equal(X_skfp, X_rdkit)


def test_mhfp_sparse_raw_hashes_fingerprint(smiles_list, mols_list):
    mhfp_fp = MHFPFingerprint(output_raw_hashes=True, sparse=True, n_jobs=-1)
    X_skfp = mhfp_fp.transform(smiles_list)

    encoder = MHFPEncoder(2048, 0)
    X_rdkit = MHFPEncoder.EncodeMolsBulk(
        encoder,
        mols_list,
    )
    X_rdkit = csr_array(X_rdkit)

    assert np.array_equal(X_skfp.data, X_rdkit.data)
