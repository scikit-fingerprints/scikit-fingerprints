import pytest
from rdkit.Chem import MolFromSmiles

from skfp.descriptors import burden


@pytest.fixture
def input_mols() -> dict[str, MolFromSmiles]:

    result = {}
    for name, smiles in [
        ("benzene", "C1=CC=CC=C1"),
        ("ethanol", "CCO"),
        ("butadiene", "C=CC=C"),
        ("cyclohexane", "C1CCCCC1"),
    ]:
        mol = MolFromSmiles(smiles)
        result[name] = mol
    return result


@pytest.mark.parametrize(
    "mol_name, expected_values",
    [
        ("benzene", [13.647, 10.379, 1.574, -1.694, 1.794, -1.474, 4.986, 1.718]),
        ("ethanol", [16.249, 10.908, 1.34, -1.522, 1.296, -1.568, 3.836, 0.318]),
        ("butadiene", [13.378, 10.646, 1.291, -1.442, 1.522, -1.21, 4.88, 2.148]),
        ("cyclohexane", [14.014, 10.012, 1.95, -2.052, 2.147, -1.855, 4.506, 0.504]),
    ],
)
def test_burden_descriptors(mol_name, expected_values, input_mols):
    mol = input_mols[mol_name]
    result = burden.burden_descriptors(mol)
    for res, exp in zip(result, expected_values):
        assert round(res, 3) == round(exp, 3)
