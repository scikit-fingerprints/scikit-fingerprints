import pytest
from rdkit.Chem import GetDistanceMatrix, Mol, MolFromSmiles

from skfp.descriptors import topological as top


def diverse_molecules() -> dict[str, Mol]:
    return {
        "ethane": MolFromSmiles("CC"),
        "ethanol": MolFromSmiles("CCO"),
        "carbon_dioxide": MolFromSmiles("O=C=O"),
        "benzene": MolFromSmiles("C1=CC=CC=C1"),
        "acetic_acid": MolFromSmiles("CC(=O)O"),
        "pyridine": MolFromSmiles("c1ccncc1"),
        "isobutane": MolFromSmiles("C(C)(C)C"),
        "pyrimidine": MolFromSmiles("c1cnc2ncnc12"),
    }


@pytest.mark.parametrize(
    "mol_name, expected_value",
    {
        "ethane": 1,
        "ethanol": 4,
        "carbon_dioxide": 4,
        "benzene": 27,
        "acetic_acid": 9,
        "pyridine": 27,
        "isobutane": 9,
        "pyrimidine": 55,
    }.items(),
)
def test_wiener_index(mol_name, expected_value):
    mol = diverse_molecules()[mol_name]
    distance_matrix = GetDistanceMatrix(mol)

    result = top.wiener_index(mol, distance_matrix)
    result_without_distance_matrix = top.wiener_index(mol)

    assert result == expected_value
    assert result_without_distance_matrix == expected_value


@pytest.mark.parametrize(
    "mol_name, expected_value",
    {
        "ethane": 1.0,
        "ethanol": 1.333,
        "carbon_dioxide": 1.333,
        "benzene": 1.8,
        "acetic_acid": 1.5,
        "pyridine": 1.8,
        "isobutane": 1.5,
        "pyrimidine": 1.964,
    }.items(),
)
def test_average_wiener_index(mol_name, expected_value):
    mol = diverse_molecules()[mol_name]
    distance_matrix = GetDistanceMatrix(mol)

    result = top.average_wiener_index(mol, distance_matrix)
    result_without_distance_matrix = top.average_wiener_index(mol)

    assert round(result, 3) == expected_value
    assert round(result_without_distance_matrix, 3) == expected_value


@pytest.mark.parametrize(
    "mol_name, expected_value",
    {
        "ethane": 1,
        "ethanol": 8,
        "carbon_dioxide": 8,
        "benzene": 261,
        "acetic_acid": 45,
        "pyridine": 261,
        "isobutane": 45,
        "pyrimidine": 997,
    }.items(),
)
def test_graph_distance_index(mol_name, expected_value):
    mol = diverse_molecules()[mol_name]
    distance_matrix = GetDistanceMatrix(mol)

    result = top.graph_distance_index(mol, distance_matrix)
    result_without_distance_matrix = top.graph_distance_index(mol)

    assert result == expected_value
    assert result_without_distance_matrix == expected_value


@pytest.mark.parametrize(
    "mol_name, expected_value",
    {
        "ethane": 2,
        "ethanol": 6,
        "carbon_dioxide": 6,
        "benzene": 24,
        "acetic_acid": 12,
        "pyridine": 24,
        "isobutane": 12,
        "pyrimidine": 42,
    }.items(),
)
def test_zagreb_index(mol_name, expected_value):
    mol = diverse_molecules()[mol_name]

    result = top.zagreb_index(mol)

    assert result == expected_value


@pytest.mark.parametrize(
    "mol_name, expected_value",
    {
        "ethane": 0,
        "ethanol": 0,
        "carbon_dioxide": 0,
        "benzene": 3,
        "acetic_acid": 0,
        "pyridine": 3,
        "isobutane": 0,
        "pyrimidine": 6,
    }.items(),
)
def test_polarity_number(mol_name, expected_value):
    mol = diverse_molecules()[mol_name]
    distance_matrix = GetDistanceMatrix(mol)

    result = top.polarity_number(mol, distance_matrix)
    result_without_distance_matrix = top.polarity_number(mol)

    assert result == expected_value
    assert result_without_distance_matrix == expected_value


@pytest.mark.parametrize(
    "mol_name, expected_value",
    {
        "benzene": 3,
        "acetic_acid": 0,
        "pyridine": 2,
        "isobutane": 0,
        "pyrimidine": 1,
    }.items(),
)
def test_polarity_number_carbon_only(mol_name, expected_value):
    mol = diverse_molecules()[mol_name]
    distance_matrix = GetDistanceMatrix(mol)

    result = top.polarity_number(mol, distance_matrix, carbon_only=True)

    assert result == expected_value


def test_polarity_number_no_carbon():
    mol = MolFromSmiles("N#N")  # Dinitrogen
    distance_matrix = GetDistanceMatrix(mol)

    with pytest.raises(
        ValueError,
        match="The molecule contains no carbon atoms, so carbon-only filtering is not possible.",
    ):
        top.polarity_number(mol, distance_matrix=distance_matrix, carbon_only=True)
