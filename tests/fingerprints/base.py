import numpy as np
import pytest
from sklearn.utils._param_validation import InvalidParameterError

from skfp.fingerprints import AtomPairFingerprint, MACCSFingerprint
from skfp.fingerprints.base import FingerprintTransformer

"""
We cannot test most of FingerprintTransformer directly, as it is an abstract base
class (ABC), but its methods are used extensively by inheriting classes. Therefore,
we use inheriting fingerprints as proxies.
"""


def test_base_is_always_fitted(smiles_list):
    atom_pair_fp = AtomPairFingerprint(n_jobs=-1)
    assert atom_pair_fp.__sklearn_is_fitted__()


def test_base_transform_copy(smiles_list):
    atom_pair_fp = AtomPairFingerprint(n_jobs=-1)
    X_skfp = atom_pair_fp.transform(smiles_list, copy=False)
    X_skfp_2 = atom_pair_fp.transform(smiles_list, copy=True)
    assert np.array_equal(X_skfp, X_skfp_2)


def test_base_invalid_params(smiles_list):
    maccs_fp = MACCSFingerprint(sparse=None)
    with pytest.raises(InvalidParameterError):
        maccs_fp.transform(smiles_list)


def test_base_hash_fingerprint_bits():
    X = [1, 2, 3, 4]
    with pytest.raises(ValueError) as exc_info:
        FingerprintTransformer._hash_fingerprint_bits(
            X, fp_size=1, count=False, sparse=False
        )

    assert "Fingerprint hashing requires instances of one of" in str(exc_info)
