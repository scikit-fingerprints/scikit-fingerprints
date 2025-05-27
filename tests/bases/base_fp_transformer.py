import numpy as np
import pytest
from rdkit.DataStructs import (
    IntSparseIntVect,
    LongSparseIntVect,
    SparseBitVect,
    UIntSparseIntVect,
    ULongSparseIntVect,
)
from sklearn.utils._param_validation import InvalidParameterError

from skfp.bases.base_fp_transformer import BaseFingerprintTransformer
from skfp.fingerprints import AtomPairFingerprint, ECFPFingerprint, MACCSFingerprint

"""
We cannot test most of BaseFingerprintTransformer directly, as it is an abstract base class (ABC),
but its methods are used extensively by inheriting classes. Therefore, we use
inheriting classes as proxies.
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
    maccs_fp = MACCSFingerprint(sparse=None)  # type: ignore
    with pytest.raises(InvalidParameterError):
        maccs_fp.transform(smiles_list)


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_base_verbose(n_jobs, smiles_list, capsys):
    ecfp_fp = ECFPFingerprint(n_jobs=n_jobs, verbose=True)
    ecfp_fp.transform(smiles_list)

    output = capsys.readouterr().err
    assert "100%" in output
    assert "it/s" in output


def test_base_hash_fingerprint_bits_right_types():
    # those types just should not raise any errors
    for fp_type in [
        IntSparseIntVect,
        LongSparseIntVect,
        SparseBitVect,
        UIntSparseIntVect,
        ULongSparseIntVect,
    ]:
        fp = fp_type(10)
        fp[0] = 1
        BaseFingerprintTransformer._hash_fingerprint_bits(
            [fp], fp_size=5, count=False, sparse=False
        )


def test_base_hash_fingerprint_bits_wrong_type():
    X = [1, 2, 3, 4]
    with pytest.raises(ValueError) as exc_info:
        BaseFingerprintTransformer._hash_fingerprint_bits(
            X, fp_size=1, count=False, sparse=False
        )

    assert "Fingerprint hashing requires instances of one of" in str(exc_info)
