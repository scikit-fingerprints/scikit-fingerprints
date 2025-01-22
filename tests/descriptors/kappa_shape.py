import pytest
from rdkit.Chem import Mol, MolFromSmiles

from skfp.descriptors import kappa_shape


@pytest.fixture
def input_mols() -> dict[str, Mol]:
    result = {}
    for name, smiles in [
        ("benzene", "C1=CC=CC=C1"),
        ("ethanol", "CCO"),
        ("propane", "CCC"),
        ("butadiene", "C=CC=C"),
        ("cyclohexane", "C1CCCCC1"),
        ("methane", "C"),
    ]:
        mol = MolFromSmiles(smiles)
        result[name] = mol
    return result


@pytest.mark.parametrize(
    "mol_name, expected_values",
    [
        ("benzene", [3.412, 1.606, 0.582]),
        ("ethanol", [2.96, 1.96, 1.96]),
        ("propane", [3.0, 2.0, 0.0]),
        ("butadiene", [3.48, 2.48, 1.48]),
        ("cyclohexane", [4.167, 2.222, 1.0]),
        ("methane", [0.0, 0.0, 0.0]),
    ],
)
def test_kappa_shape_indices(mol_name, expected_values, input_mols):
    mol = input_mols[mol_name]
    result = kappa_shape.kappa_shape_indices(mol)
    assert len(result) == len(expected_values)
    for res, exp in zip(result, expected_values):
        assert round(res, 3) == round(exp, 3)
