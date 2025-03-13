import pytest
from sklearn.utils._param_validation import InvalidParameterError

from skfp.preprocessing import MolFromSmilesTransformer

"""
We cannot test most of BasePreprocessor directly, as it is an abstract base class (ABC),
but its methods are used extensively by inheriting classes. Therefore, we use
inheriting classes as proxies.
"""


def test_base_is_always_fitted(smiles_list):
    mol_from_smiles = MolFromSmilesTransformer()
    assert mol_from_smiles.__sklearn_is_fitted__()


def test_base_invalid_params(smiles_list):
    mol_from_smiles = MolFromSmilesTransformer(sanitize=-1)  # type: ignore
    with pytest.raises(InvalidParameterError):
        mol_from_smiles.transform(smiles_list)
