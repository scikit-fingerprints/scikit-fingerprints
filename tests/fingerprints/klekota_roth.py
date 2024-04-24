import numpy as np
import pytest
from scipy.sparse import csr_array
from sklearn.utils._param_validation import InvalidParameterError

from skfp.fingerprints import KlekotaRothFingerprint


def test_klekota_roth_bit_fingerprint(smiles_list):
    fp = KlekotaRothFingerprint(sparse=False, count=False, n_jobs=-1)
    X = fp.transform(smiles_list)

    assert type(X) is np.ndarray
    assert X.dtype == np.uint8
    assert X.shape == (len(smiles_list), 4860)
    assert np.all(np.isin(X, [0, 1]))


def test_klekota_roth_count_fingerprint(smiles_list):
    fp = KlekotaRothFingerprint(sparse=False, count=True, n_jobs=-1)
    X = fp.transform(smiles_list)

    assert type(X) is np.ndarray
    assert X.dtype == np.uint8
    assert X.shape == (len(smiles_list), 4860)
    assert np.all(X >= 0)


def test_parameter_constraints_enabled(mols_list):
    with pytest.raises(InvalidParameterError):
        fp = KlekotaRothFingerprint(count=42)  # type: ignore
        fp.transform(mols_list)


# def test_substructure_validation():
#     invalid_substructures = [1, True, "abc"]
#     with pytest.raises(InvalidParameterError):
#         KlekotaRothFingerprint(invalid_substructures)
