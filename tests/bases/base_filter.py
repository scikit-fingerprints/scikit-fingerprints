import pytest
from sklearn.utils._param_validation import InvalidParameterError

from skfp.filters import LipinskiFilter

"""
We cannot test most of BaseFilter directly, as it is an abstract base class (ABC),
but its methods are used extensively by inheriting classes. Therefore, we use
inheriting classes as proxies.
"""


def test_base_is_always_fitted(smiles_list):
    filt = LipinskiFilter()
    assert filt.__sklearn_is_fitted__()


def test_base_transform_copy(smiles_list):
    filt = LipinskiFilter()
    filtered_smiles = filt.transform(smiles_list, copy=False)
    filtered_smiles_2 = filt.transform(smiles_list, copy=True)
    assert filtered_smiles == filtered_smiles_2


def test_base_invalid_params(smiles_list):
    filt = LipinskiFilter(allow_one_violation=2)  # type: ignore
    with pytest.raises(InvalidParameterError):
        filt.transform(smiles_list)
