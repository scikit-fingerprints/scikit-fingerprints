import numpy as np
import pytest
from rdkit.Chem.rdmolops import LayeredFingerprint as RDKitLayeredFingerprint
from scipy.sparse import csr_array
from sklearn.utils._param_validation import InvalidParameterError

from skfp.fingerprints import LayeredFingerprint


def test_layered_fingerprint(smiles_list, mols_list):
    layered_fp = LayeredFingerprint(n_jobs=-1)
    X_skfp = layered_fp.transform(smiles_list)
    X_rdkit = np.array([RDKitLayeredFingerprint(mol) for mol in mols_list])

    assert np.array_equal(X_skfp, X_rdkit)
    assert X_skfp.shape == (len(smiles_list), layered_fp.fp_size)
    assert X_skfp.dtype == np.uint8
    assert np.all(np.isin(X_skfp, [0, 1]))


def test_layered_sparse_fingerprint(smiles_list, mols_list):
    layered_fp = LayeredFingerprint(sparse=True, n_jobs=-1)
    X_skfp = layered_fp.transform(smiles_list)
    X_rdkit = csr_array([RDKitLayeredFingerprint(mol) for mol in mols_list])

    assert np.array_equal(X_skfp.data, X_rdkit.data)
    assert X_skfp.shape == (len(smiles_list), layered_fp.fp_size)
    assert X_skfp.dtype == np.uint8
    assert np.all(X_skfp.data == 1)


def test_layered_wrong_path_lengths(smiles_list):
    layered_fp = LayeredFingerprint(min_path=3, max_path=2)
    with pytest.raises(InvalidParameterError):
        layered_fp.transform(smiles_list)
