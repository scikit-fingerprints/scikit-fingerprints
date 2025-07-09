from collections.abc import Sequence

import pytest
from rdkit.Chem import Mol, MolToSmiles
from sklearn.utils._param_validation import InvalidParameterError

from skfp.preprocessing import (
    MolFromAminoseqTransformer,
    MolFromInchiTransformer,
    MolFromSmilesTransformer,
)

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
    assert all(isinstance(elem, Mol) for elem in results)


@pytest.mark.parametrize(
    "transformer_cls, inputs",
    [
        (MolFromSmilesTransformer, ["C", "CC", "CCC"]),
        (
            MolFromInchiTransformer,
            [
                "InChI=1S/CH4/h1H4",
                "InChI=1S/C2H6/c1-2/h1-2H3",
            ],
        ),
        (
            MolFromAminoseqTransformer,
            ["ACDEFGHIKLMNPQRSTVWY", "MEEPQSDPSVEPPLSQETFSDLWKLL"],
        ),
    ],
)
@pytest.mark.parametrize("verbose", [True, False])
def test_sequential_and_parallel_results(transformer_cls, inputs, verbose):
    transformer_seq = transformer_cls(n_jobs=1, verbose=verbose)
    transformer_par = transformer_cls(n_jobs=2, verbose=verbose)

    results_seq = transformer_seq.transform(inputs)
    results_par = transformer_par.transform(inputs)

    seq_smiles = [MolToSmiles(mol) for mol in results_seq]
    par_smiles = [MolToSmiles(mol) for mol in results_par]

    assert seq_smiles == par_smiles
