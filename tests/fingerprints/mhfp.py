import numpy as np
import pytest
from rdkit.Chem.rdMHFPFingerprint import MHFPEncoder
from scipy.sparse import csr_array
from sklearn.utils._param_validation import InvalidParameterError

from skfp.fingerprints import MHFPFingerprint


def test_mhfp_bit_fingerprint(smiles_list, mols_list):
    mhfp_fp = MHFPFingerprint(variant="bit", sparse=False, n_jobs=-1)
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
    assert X_skfp.shape == (len(smiles_list), mhfp_fp.fp_size)


def test_mhfp_count_fingerprint(smiles_list, mols_list):
    mhfp_fp = MHFPFingerprint(variant="count", sparse=False, n_jobs=-1)
    X_skfp = mhfp_fp.transform(smiles_list)

    encoder = MHFPEncoder(2048, 0)
    X_rdkit = MHFPEncoder.EncodeMolsBulk(
        encoder,
        mols_list,
    )
    X_rdkit = np.array(X_rdkit)
    X_rdkit = np.mod(X_rdkit, mhfp_fp.fp_size)
    X_rdkit = np.stack([np.bincount(x, minlength=mhfp_fp.fp_size) for x in X_rdkit])

    assert np.array_equal(X_skfp, X_rdkit)
    assert X_skfp.shape == (len(smiles_list), mhfp_fp.fp_size)


def test_mhfp_raw_hashes_fingerprint(smiles_list, mols_list):
    mhfp_fp = MHFPFingerprint(variant="raw_hashes", sparse=False, n_jobs=-1)
    X_skfp = mhfp_fp.transform(smiles_list)

    encoder = MHFPEncoder(2048, 0)
    X_rdkit = MHFPEncoder.EncodeMolsBulk(
        encoder,
        mols_list,
    )
    X_rdkit = np.array(X_rdkit)

    assert np.array_equal(X_skfp, X_rdkit)
    assert X_skfp.shape == (len(smiles_list), mhfp_fp.fp_size)


def test_mhfp_sparse_bit_fingerprint(smiles_list, mols_list):
    mhfp_fp = MHFPFingerprint(variant="bit", sparse=True, n_jobs=-1)
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
    assert X_skfp.shape == (len(smiles_list), mhfp_fp.fp_size)


def test_mhfp_sparse_count_fingerprint(smiles_list, mols_list):
    mhfp_fp = MHFPFingerprint(variant="count", sparse=True, n_jobs=-1)
    X_skfp = mhfp_fp.transform(smiles_list)

    encoder = MHFPEncoder(2048, 0)
    X_rdkit = MHFPEncoder.EncodeMolsBulk(
        encoder,
        mols_list,
    )
    X_rdkit = np.array(X_rdkit)
    X_rdkit = np.mod(X_rdkit, mhfp_fp.fp_size)
    X_rdkit = csr_array(
        [np.bincount(x, minlength=mhfp_fp.fp_size) for x in X_rdkit],
        dtype=int,
    )

    assert np.array_equal(X_skfp.data, X_rdkit.data)
    assert X_skfp.shape == (len(smiles_list), mhfp_fp.fp_size)


def test_mhfp_sparse_raw_hashes_fingerprint(smiles_list, mols_list):
    mhfp_fp = MHFPFingerprint(variant="raw_hashes", sparse=True, n_jobs=-1)
    X_skfp = mhfp_fp.transform(smiles_list)

    encoder = MHFPEncoder(2048, 0)
    X_rdkit = MHFPEncoder.EncodeMolsBulk(
        encoder,
        mols_list,
    )
    X_rdkit = csr_array(X_rdkit)

    assert np.array_equal(X_skfp.data, X_rdkit.data)
    assert X_skfp.shape == (len(smiles_list), mhfp_fp.fp_size)


def test_mhfp_wrong_radii(smiles_list):
    mhfp_fp = MHFPFingerprint(min_radius=3, radius=2)
    with pytest.raises(InvalidParameterError):
        mhfp_fp.transform(smiles_list)
