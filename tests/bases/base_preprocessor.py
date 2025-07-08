from collections.abc import Sequence

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


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_base_verbose(n_jobs, smiles_list, capsys):
    mol_from_smiles = MolFromSmilesTransformer(n_jobs=n_jobs, verbose=True)
    mol_from_smiles.transform(smiles_list)

    output = capsys.readouterr().err
    assert "100%" in output
    assert "it/s" in output


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("verbose", [True, False])
def test_base_flattened_results(n_jobs, verbose, smiles_list):
    mol_from_smiles = MolFromSmilesTransformer(n_jobs=n_jobs, verbose=verbose)
    results = mol_from_smiles.transform(smiles_list)

    assert isinstance(results, Sequence)
    assert len(results) == len(smiles_list)

    for x in results:
        if isinstance(x, Sequence):
            raise AssertionError(f"Nested sequence detected {x}")
