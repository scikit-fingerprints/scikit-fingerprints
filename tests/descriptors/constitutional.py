import pytest
from rdkit.Chem import Mol, MolFromSmiles

from skfp.descriptors import constitutional as const


@pytest.fixture
def input_mols() -> dict[str, Mol]:
    result = {}
    for name, smiles in [
        ("oxygen", "O"),
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
    "mol_name, expected_value",
    [
        ("benzene", 13.019),
        ("ethanol", 15.356),
        ("propane", 14.699),
        ("butadiene", 13.523),
        ("hydrogen_cyanide", 13.513),
        ("cyclohexane", 14.027),
        ("oxygen", 18.015),
    ],
)
def test_average_molecular_weight(mol_name, expected_value, input_mols):
    mol = input_mols[mol_name]
    result = const.average_molecular_weight(mol)
    assert round(result, 3) == round(expected_value, 3)


@pytest.mark.parametrize(
    "mol_name, bond_type, expected_value",
    [
        ("benzene", "SINGLE", 0),
        ("benzene", "DOUBLE", 0),
        ("benzene", "TRIPLE", 0),
        ("benzene", "AROMATIC", 6),
        ("ethanol", "SINGLE", 2),
        ("ethanol", "DOUBLE", 0),
        ("ethanol", "TRIPLE", 0),
        ("butadiene", "DOUBLE", 2),
        ("hydrogen_cyanide", "TRIPLE", 1),
        ("cyclohexane", "SINGLE", 6),
    ],
)
def test_bond_type_count(mol_name, bond_type, expected_value, input_mols):
    mol = input_mols[mol_name]
    result = const.bond_type_count(mol, bond_type)
    assert result == expected_value


@pytest.mark.parametrize(
    "mol_name, atom_id, expected_value",
    [
        ("benzene", "C", 6),
        ("ethanol", "O", 1),
        ("propane", "H", 8),
        ("hydrogen_cyanide", "N", 1),
        ("butadiene", 1, 6),
        ("ethanol", 6, 2),
        ("oxygen", 8, 1),
    ],
)
def test_element_atom_count(mol_name, atom_id, expected_value, input_mols):
    mol = input_mols[mol_name]
    result = const.element_atom_count(mol, atom_id)
    assert result == expected_value


@pytest.mark.parametrize(
    "mol_name, expected_value",
    [
        ("benzene", 6),
        ("ethanol", 3),
        ("propane", 3),
        ("butadiene", 4),
        ("hydrogen_cyanide", 2),
        ("cyclohexane", 6),
        ("oxygen", 1),
    ],
)
def test_heavy_atom_count(mol_name, expected_value, input_mols):
    mol = input_mols[mol_name]
    result = const.heavy_atom_count(mol)
    assert result == expected_value


@pytest.mark.parametrize(
    "mol_name, expected_value",
    [
        ("benzene", 78.114),
        ("ethanol", 46.069),
        ("propane", 44.097),
        ("butadiene", 54.092),
        ("hydrogen_cyanide", 27.026),
        ("cyclohexane", 84.162),
        ("oxygen", 18.015),
    ],
)
def test_molecular_weight(mol_name, expected_value, input_mols):
    mol = input_mols[mol_name]
    result = const.molecular_weight(mol)
    assert round(result, 3) == round(expected_value, 3)


@pytest.mark.parametrize(
    "mol_name, expected_value",
    [
        ("benzene", 1),
        ("cyclohexane", 1),
        ("ethanol", 0),
        ("hydrogen_cyanide", 0),
        ("butadiene", 0),
        ("oxygen", 0),
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
        ("oxygen", 0),
    ],
)
def test_number_of_rotatable_bonds(mol_name, expected_value, input_mols):
    mol = input_mols[mol_name]
    result = const.number_of_rotatable_bonds(mol)
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
        ("oxygen", 3),
    ],
)
def test_total_atom_count(mol_name, expected_value, input_mols):
    mol = input_mols[mol_name]
    result = const.total_atom_count(mol)
    assert result == expected_value


@pytest.mark.parametrize(
    "descriptor_function",
    [
        const.average_molecular_weight,
        const.bond_type_count,
        const.element_atom_count,
        const.heavy_atom_count,
        const.molecular_weight,
        const.number_of_rings,
        const.number_of_rotatable_bonds,
        const.total_atom_count,
    ],
)
def test_empty_molecule_raises_error(descriptor_function):
    mol = MolFromSmiles("")
    with pytest.raises(ValueError, match="The molecule has no atoms"):
        descriptor_function(mol)
