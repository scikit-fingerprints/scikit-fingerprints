import pytest
from rdkit.Chem import MolFromSmiles, Mol, GetDistanceMatrix

from skfp.descriptors import topological as top


def diverse_molecules() -> list[tuple[str, Mol]]:
    return [
        ("ethane", MolFromSmiles("CC")),
        ("ethanol", MolFromSmiles("CCO")),
        ("carbon_dioxide", MolFromSmiles("O=C=O")),
        ("benzene", MolFromSmiles("C1=CC=CC=C1")),
        ("acetic_acid", MolFromSmiles("CC(=O)O")),
        ("pyridine", MolFromSmiles("c1ccncc1")),
        ("isobutane", MolFromSmiles("C(C)(C)C")),
        ("pyrimidine", MolFromSmiles("c1cnc2ncnc12")),
    ]


@pytest.mark.parametrize(
    "mol_name, expected_value",
    [
        ("ethane", 1),
        ("ethanol", 4),
        ("carbon_dioxide", 4),
        ("benzene", 27),
        ("acetic_acid", 9),
        ("pyridine", 27),
        ("isobutane", 9),
        ("pyrimidine", 55),
    ],
)
def test_weiner_index(mol_name, expected_value):
    mol_tuple = next(item for item in diverse_molecules() if item[0] == mol_name)
    mol = mol_tuple[1]

    distance_matrix = GetDistanceMatrix(mol)

    result = top.weiner_index(mol, distance_matrix)
    result_without_distance_matrix = top.weiner_index(mol)

    assert result == pytest.approx(expected_value, abs=1e-6)
    assert result_without_distance_matrix == pytest.approx(expected_value, abs=1e-6)


@pytest.mark.parametrize(
    "mol_name, expected_value",
    [
        ("ethane", 1.0),
        ("ethanol", 1.3333333333333333),
        ("carbon_dioxide", 1.3333333333333333),
        ("benzene", 1.8),
        ("acetic_acid", 1.5),
        ("pyridine", 1.8),
        ("isobutane", 1.5),
        ("pyrimidine", 1.9642857142857142),
    ],
)
def test_average_weiner_index(mol_name, expected_value):
    mol_tuple = next(item for item in diverse_molecules() if item[0] == mol_name)
    mol = mol_tuple[1]

    distance_matrix = GetDistanceMatrix(mol)

    result = top.average_weiner_index(mol, distance_matrix)
    result_without_distance_matrix = top.average_weiner_index(mol)

    assert result == pytest.approx(expected_value, abs=1e-6)
    assert result_without_distance_matrix == pytest.approx(expected_value, abs=1e-6)


@pytest.mark.parametrize(
    "mol_name, expected_value",
    [
        ("ethane", 1.0),
        ("ethanol", 8.0),
        ("carbon_dioxide", 8.0),
        ("benzene", 261.0),
        ("acetic_acid", 45.0),
        ("pyridine", 261.0),
        ("isobutane", 45.0),
        ("pyrimidine", 997.0),
    ],
)
def test_graph_distance_index(mol_name, expected_value):
    mol_tuple = next(item for item in diverse_molecules() if item[0] == mol_name)
    mol = mol_tuple[1]

    distance_matrix = GetDistanceMatrix(mol)

    result = top.graph_distance_index(mol, distance_matrix)
    result_without_distance_matrix = top.graph_distance_index(mol)

    assert result == pytest.approx(expected_value, abs=1e-6)
    assert result_without_distance_matrix == pytest.approx(expected_value, abs=1e-6)


@pytest.mark.parametrize(
    "mol_name, expected_value",
    [
        ("ethane", 2),
        ("ethanol", 6),
        ("carbon_dioxide", 6),
        ("benzene", 24),
        ("acetic_acid", 12),
        ("pyridine", 24),
        ("isobutane", 12),
        ("pyrimidine", 42),
    ],
)
def test_zagreb_index(mol_name, expected_value):
    mol_tuple = next(item for item in diverse_molecules() if item[0] == mol_name)
    mol = mol_tuple[1]

    result = top.zagreb_index(mol)

    assert result == pytest.approx(expected_value, abs=1e-6)


@pytest.mark.parametrize(
    "mol_name, expected_value",
    [
        ("ethane", 0),
        ("ethanol", 0),
        ("carbon_dioxide", 0),
        ("benzene", 3),
        ("acetic_acid", 0),
        ("pyridine", 3),
        ("isobutane", 0),
        ("pyrimidine", 6),
    ],
)
def test_polarity_number(mol_name, expected_value):
    mol_tuple = next(item for item in diverse_molecules() if item[0] == mol_name)
    mol = mol_tuple[1]

    distance_matrix = GetDistanceMatrix(mol)

    result = top.polarity_number(mol, distance_matrix)
    result_without_distance_matrix = top.polarity_number(mol)

    assert result == pytest.approx(expected_value, abs=1e-6)
    assert result_without_distance_matrix == pytest.approx(expected_value, abs=1e-6)


@pytest.mark.parametrize(
    "mol_name, expected_value",
    [
        ("benzene", 3),
        ("acetic_acid", 0),
        ("pyridine", 2),
        ("isobutane", 0),
        ("pyrimidine", 1),
    ],
)
def test_polarity_number_carbon_only(mol_name, expected_value):
    mol_tuple = next(item for item in diverse_molecules() if item[0] == mol_name)
    mol = mol_tuple[1]

    distance_matrix = GetDistanceMatrix(mol)

    result = top.polarity_number(mol, distance_matrix, carbon_only=True)

    assert (result == pytest.approx(expected_value, abs=1e-6))
