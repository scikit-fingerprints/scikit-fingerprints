import numpy as np
import pytest
from sklearn.utils._param_validation import InvalidParameterError

from skfp.fingerprints import KlekotaRothFingerprint


def test_klekota_roth_bit_fingerprint(smiles_list):
    fp = KlekotaRothFingerprint(sparse=False, count=False, n_jobs=-1)
    X = fp.transform(smiles_list)

    assert isinstance(X, np.ndarray)
    assert X.dtype == np.uint8
    assert X.shape == (len(smiles_list), 4860)
    assert np.all(np.isin(X, [0, 1]))


def test_klekota_roth_count_fingerprint(smiles_list):
    fp = KlekotaRothFingerprint(sparse=False, count=True, n_jobs=-1)
    X = fp.transform(smiles_list)

    assert isinstance(X, np.ndarray)
    assert X.dtype == np.uint32
    assert X.shape == (len(smiles_list), 4860)
    assert np.all(X >= 0)


def test_parameter_constraints_enabled(mols_list):
    with pytest.raises(InvalidParameterError) as error:
        fp = KlekotaRothFingerprint(count=42)  # type: ignore
        fp.transform(mols_list)

    assert str(error.value).startswith(
        "The 'count' parameter of KlekotaRothFingerprint must be an instance of 'bool'"
    )
