import numpy as np
from descriptastorus.descriptors import RDKit2D, RDKit2DNormalized
from scipy.sparse import csr_array

from skfp.fingerprints import RDKit2DDescriptorsFingerprint
from skfp.utils import no_rdkit_logs


def test_rdkit_2d_desc_fingerprint(smallest_mols_list):
    getaway_fp = RDKit2DDescriptorsFingerprint(n_jobs=-1)
    X_skfp = getaway_fp.transform(smallest_mols_list)

    gen = RDKit2D()
    with no_rdkit_logs():
        X_descriptastorus = [
            np.array(gen.calculateMol(mol, None)) for mol in smallest_mols_list
        ]
    X_descriptastorus = [np.clip(x, -2147483647, 2147483647) for x in X_descriptastorus]
    X_descriptastorus = np.array(X_descriptastorus, dtype=np.float32)
    colnames = getaway_fp.get_feature_names_out()

    assert np.allclose(X_skfp, X_descriptastorus, atol=1e-3, equal_nan=True)
    assert X_skfp.shape == (len(smallest_mols_list), 200)
    assert np.issubdtype(X_skfp.dtype, np.floating)
    assert np.array_equal(colnames, list(zip(*gen.columns))[0])


def test_rdkit_2d_desc_sparse_fingerprint(smallest_mols_list):
    getaway_fp = RDKit2DDescriptorsFingerprint(sparse=True, n_jobs=-1)
    X_skfp = getaway_fp.transform(smallest_mols_list)

    gen = RDKit2D()
    with no_rdkit_logs():
        X_descriptastorus = [
            np.array(gen.calculateMol(mol, None)) for mol in smallest_mols_list
        ]
    X_descriptastorus = [np.clip(x, -2147483647, 2147483647) for x in X_descriptastorus]
    X_descriptastorus = csr_array(X_descriptastorus)
    colnames = getaway_fp.get_feature_names_out()

    assert np.allclose(X_skfp.data, X_descriptastorus.data, atol=1e-3, equal_nan=True)  # type: ignore
    assert X_skfp.shape == (len(smallest_mols_list), 200)
    assert np.issubdtype(X_skfp.dtype, np.floating)
    assert np.array_equal(colnames, list(zip(*gen.columns))[0])


def test_rdkit_2d_desc_normalized_fingerprint(smallest_mols_list):
    getaway_fp = RDKit2DDescriptorsFingerprint(normalized=True, n_jobs=-1)
    X_skfp = getaway_fp.transform(smallest_mols_list)

    gen = RDKit2DNormalized()
    with no_rdkit_logs():
        X_descriptastorus = [
            np.array(gen.calculateMol(mol, None)) for mol in smallest_mols_list
        ]
    X_descriptastorus = [np.clip(x, -2147483647, 2147483647) for x in X_descriptastorus]
    X_descriptastorus = np.vstack(X_descriptastorus)
    colnames = getaway_fp.get_feature_names_out()

    assert np.allclose(X_skfp, X_descriptastorus, atol=1e-3, equal_nan=True)
    assert X_skfp.shape == (len(smallest_mols_list), 200)
    assert np.issubdtype(X_skfp.dtype, np.floating)
    assert np.array_equal(colnames, list(zip(*gen.columns))[0])


def test_rdkit_2d_desc_normalized_sparse_fingerprint(smallest_mols_list):
    getaway_fp = RDKit2DDescriptorsFingerprint(normalized=True, sparse=True, n_jobs=-1)
    X_skfp = getaway_fp.transform(smallest_mols_list)

    gen = RDKit2DNormalized()
    with no_rdkit_logs():
        X_descriptastorus = [
            np.array(gen.calculateMol(mol, None)) for mol in smallest_mols_list
        ]
    X_descriptastorus = [np.clip(x, -2147483647, 2147483647) for x in X_descriptastorus]
    X_descriptastorus = csr_array(X_descriptastorus)
    colnames = getaway_fp.get_feature_names_out()

    assert np.allclose(X_skfp.data, X_descriptastorus.data, atol=1e-3, equal_nan=True)  # type: ignore
    assert X_skfp.shape == (len(smallest_mols_list), 200)
    assert np.issubdtype(X_skfp.dtype, np.floating)
    assert np.array_equal(colnames, list(zip(*gen.columns))[0])
