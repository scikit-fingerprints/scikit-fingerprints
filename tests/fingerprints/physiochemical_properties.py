import numpy as np
from scipy.sparse import csr_array

from skfp.fingerprints import PhysiochemicalPropertiesFingerprint


def test_physiochemical_properties_bp_bit_fingerprint(smiles_list):
    pp_fp = PhysiochemicalPropertiesFingerprint(variant="BP", n_jobs=-1)
    X_skfp = pp_fp.transform(smiles_list)

    assert isinstance(X_skfp, np.ndarray)
    assert X_skfp.shape == (len(smiles_list), pp_fp.fp_size)
    assert X_skfp.dtype == np.uint8
    assert np.all(np.isin(X_skfp, [0, 1]))


def test_physiochemical_properties_bp_count_fingerprint(smiles_list):
    pp_fp = PhysiochemicalPropertiesFingerprint(variant="BP", count=True, n_jobs=-1)
    X_skfp = pp_fp.transform(smiles_list)

    assert isinstance(X_skfp, np.ndarray)
    assert X_skfp.shape == (len(smiles_list), pp_fp.fp_size)
    assert X_skfp.dtype == np.uint32
    assert np.all(X_skfp >= 0)


def test_physiochemical_properties_bt_bit_fingerprint(smiles_list):
    pp_fp = PhysiochemicalPropertiesFingerprint(variant="BT", n_jobs=-1)
    X_skfp = pp_fp.transform(smiles_list)

    assert isinstance(X_skfp, np.ndarray)
    assert X_skfp.shape == (len(smiles_list), pp_fp.fp_size)
    assert X_skfp.dtype == np.uint8
    assert np.all(np.isin(X_skfp, [0, 1]))


def test_physiochemical_properties_bt_count_fingerprint(smiles_list):
    pp_fp = PhysiochemicalPropertiesFingerprint(variant="BT", count=True, n_jobs=-1)
    X_skfp = pp_fp.transform(smiles_list)

    assert isinstance(X_skfp, np.ndarray)
    assert X_skfp.shape == (len(smiles_list), pp_fp.fp_size)
    assert X_skfp.dtype == np.uint32
    assert np.all(X_skfp >= 0)


def test_physiochemical_properties_bp_sparse_bit_fingerprint(smiles_list):
    pp_fp = PhysiochemicalPropertiesFingerprint(variant="BP", sparse=True, n_jobs=-1)
    X_skfp = pp_fp.transform(smiles_list)

    assert isinstance(X_skfp, csr_array)
    assert X_skfp.shape == (len(smiles_list), pp_fp.fp_size)
    assert X_skfp.dtype == np.uint8
    assert np.all(X_skfp.data == 1)


def test_physiochemical_properties_bp_sparse_count_fingerprint(smiles_list):
    pp_fp = PhysiochemicalPropertiesFingerprint(
        variant="BP", sparse=True, count=True, n_jobs=-1
    )
    X_skfp = pp_fp.transform(smiles_list)

    assert isinstance(X_skfp, csr_array)
    assert X_skfp.shape == (len(smiles_list), pp_fp.fp_size)
    assert X_skfp.dtype == np.uint32
    assert np.all(X_skfp.data > 0)


def test_physiochemical_properties_bt_sparse_bit_fingerprint(smiles_list):
    pp_fp = PhysiochemicalPropertiesFingerprint(variant="BT", sparse=True, n_jobs=-1)
    X_skfp = pp_fp.transform(smiles_list)

    assert isinstance(X_skfp, csr_array)
    assert X_skfp.shape == (len(smiles_list), pp_fp.fp_size)
    assert X_skfp.dtype == np.uint8
    assert np.all(X_skfp.data == 1)


def test_physiochemical_properties_bt_sparse_count_fingerprint(smiles_list):
    pp_fp = PhysiochemicalPropertiesFingerprint(
        variant="BT", sparse=True, count=True, n_jobs=-1
    )
    X_skfp = pp_fp.transform(smiles_list)

    assert isinstance(X_skfp, csr_array)
    assert X_skfp.shape == (len(smiles_list), pp_fp.fp_size)
    assert X_skfp.dtype == np.uint32
    assert np.all(X_skfp.data > 0)
