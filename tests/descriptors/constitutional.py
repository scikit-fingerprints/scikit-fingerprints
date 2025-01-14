import pytest
from rdkit.Chem import Mol, MolFromSmiles

from skfp.descriptors import constitutional as const


@pytest.fixture
def input_mols() -> dict[str, Mol]:
    result = {}
    for name, smiles in [
        ("benzene", "C1=CC=CC=C1"),
        ("ethanol", "CCO"),
        ("propane", "CCC"),
        ("butadiene", "C=CC=C"),
        ("hydrogen_cyanide", "C#N"),
        ("cyclohexane", "C1CCCCC1"),
    ]:
        mol = MolFromSmiles(smiles)
        result[name] = mol
    return result


@pytest.mark.parametrize(
    "mol_name, atom_symbol, expected_value",
    [
        ("benzene", "C", 6),
        ("ethanol", "C", 2),
        ("ethanol", "O", 1),
        ("propane", "H", 8),
        ("hydrogen_cyanide", "N", 1),
        ("butadiene", "H", 6),
    ],
)
def test_atom_count(mol_name, atom_symbol, expected_value, input_mols):
    mol = input_mols[mol_name]
    result = const.atom_count(mol, atom_symbol)
    assert result == expected_value


@pytest.mark.parametrize(
    "mol_name, expected_value",
    [
        ("benzene", 13.019),
        ("ethanol", 15.356),
        ("propane", 14.699),
        ("butadiene", 13.523),
        ("hydrogen_cyanide", 13.513),
        ("cyclohexane", 14.027),
    ],
)
def test_average_molecular_weight(mol_name, expected_value, input_mols):
    mol = input_mols[mol_name]
    result = const.average_molecular_weight(mol)
    assert round(result, 3) == round(expected_value, 3)


@pytest.mark.parametrize(
    "mol_name, expected_value",
    [
        ("benzene", 78.114),
        ("ethanol", 46.069),
        ("propane", 44.097),
        ("butadiene", 54.092),
        ("hydrogen_cyanide", 27.026),
        ("cyclohexane", 84.162),
    ],
)
def test_molecular_weight(mol_name, expected_value, input_mols):
    mol = input_mols[mol_name]
    result = const.molecular_weight(mol)
    assert round(result, 3) == round(expected_value, 3)


@pytest.mark.parametrize(
    "mol_name, expected_value",
    [
        ("benzene", 0),
        ("ethanol", 0),
        ("propane", 0),
        ("butadiene", 2),
        ("hydrogen_cyanide", 0),
    ],
)
def test_number_of_double_bonds(mol_name, expected_value, input_mols):
    mol = input_mols[mol_name]
    result = const.number_of_double_bonds(mol)
    assert result == expected_value


@pytest.mark.parametrize(
    "mol_name, expected_value",
    [
        ("benzene", 1),
        ("cyclohexane", 1),
        ("ethanol", 0),
        ("hydrogen_cyanide", 0),
        ("butadiene", 0),
    ],
)
def test_number_of_rings(mol_name, expected_value, input_mols):
    mol = input_mols[mol_name]
    result = const.number_of_rings(mol)
    assert result == expected_value


@pytest.mark.parametrize(
    "mol_name, expected_value",
    [
        ("benzene", 0),
        ("ethanol", 0),
        ("propane", 0),
        ("hydrogen_cyanide", 0),
        ("butadiene", 1),
    ],
)
def test_number_of_rotatable_bonds(mol_name, expected_value, input_mols):
    mol = input_mols[mol_name]
    result = const.number_of_rotatable_bonds(mol)
    assert result == expected_value


@pytest.mark.parametrize(
    "mol_name, expected_value",
    [
        ("benzene", 0),
        ("ethanol", 2),
        ("propane", 2),
        ("hydrogen_cyanide", 0),
        ("butadiene", 1),
    ],
)
def test_number_of_single_bonds(mol_name, expected_value, input_mols):
    mol = input_mols[mol_name]
    result = const.number_of_single_bonds(mol)
    assert result == expected_value


@pytest.mark.parametrize(
    "mol_name, expected_value",
    [
        ("benzene", 0),
        ("ethanol", 0),
        ("propane", 0),
        ("hydrogen_cyanide", 1),
        ("butadiene", 0),
    ],
)
def test_number_of_triple_bonds(mol_name, expected_value, input_mols):
    mol = input_mols[mol_name]
    result = const.number_of_triple_bonds(mol)
    assert result == expected_value


@pytest.mark.parametrize(
    "mol_name, expected_value",
    [
        ("benzene", 12),
        ("ethanol", 9),
        ("propane", 11),
        ("butadiene", 10),
        ("hydrogen_cyanide", 3),
        ("cyclohexane", 18),
    ],
)
def test_total_atom_count(mol_name, expected_value, input_mols):
    mol = input_mols[mol_name]
    result = const.total_atom_count(mol)
    assert result == expected_value
