import numpy as np
import pytest
from rdkit.Chem import GetDistanceMatrix, Mol, MolFromSmiles

from skfp.descriptors import topological as top


@pytest.fixture
def input_mols() -> dict[str, tuple[Mol, np.ndarray]]:
    result = {}
    for name, smiles in [
        ("ethane", "CC"),
        ("ethanol", "CCO"),
        ("carbon_dioxide", "O=C=O"),
        ("benzene", "C1=CC=CC=C1"),
        ("acetic_acid", "CC(=O)O"),
        ("pyridine", "c1ccncc1"),
        ("isobutane", "C(C)(C)C"),
        ("pyrimidine", "c1cnc2ncnc12"),
        ("dinitrogen", "N#N"),
        ("sulfur", "[S]"),
    ]:
        mol = MolFromSmiles(smiles)
        dist_matrix = GetDistanceMatrix(mol)
        result[name] = (mol, dist_matrix)

    return result


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
        "sulfur": 0,
    }.items(),
)
def test_wiener_index(mol_name, expected_value, input_mols):
    mol, distance_matrix = input_mols[mol_name]

    result = top.wiener_index(mol, distance_matrix)
    result_no_dist_matrix = top.wiener_index(mol)

    assert result == expected_value
    assert result_no_dist_matrix == expected_value


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
def test_average_wiener_index(mol_name, expected_value, input_mols):
    mol, distance_matrix = input_mols[mol_name]

    result = top.average_wiener_index(mol, distance_matrix)
    result_no_dist_matrix = top.average_wiener_index(mol)

    assert round(result, 3) == expected_value
    assert round(result_no_dist_matrix, 3) == expected_value


def test_average_wiener_index_single_atom(input_mols):
    mol, distance_matrix = input_mols["sulfur"]
    with pytest.raises(
        ValueError,
        match="The molecule must have at least 2 atom\\(s\\), average_wiener_index cannot be calculated.",
    ):
        top.average_wiener_index(mol)


@pytest.mark.parametrize(
    "mol_name, expected_value",
    {
        "ethane": 1.0,
        "ethanol": 1.633,
        "carbon_dioxide": 3.266,
        "benzene": 3.0,
        "acetic_acid": 2.803,
        "pyridine": 3.0,
        "isobutane": 2.324,
        "pyrimidine": 2.591,
        "sulfur": 0,
    }.items(),
)
def test_balaban_j_index(mol_name, expected_value, input_mols):
    mol, distance_matrix = input_mols[mol_name]

    result = top.balaban_j_index(mol)
    result_no_dist_matrix = top.balaban_j_index(mol)

    assert round(result, 3) == expected_value
    assert round(result_no_dist_matrix, 3) == expected_value


@pytest.mark.parametrize(
    "mol_name, expected_value",
    {
        "ethane": 1,
        "ethanol": 2,
        "carbon_dioxide": 2,
        "benzene": 3,
        "acetic_acid": 2,
        "pyridine": 3,
        "isobutane": 2,
        "pyrimidine": 4,
        "sulfur": 0,
    }.items(),
)
def test_diameter(mol_name, expected_value, input_mols):
    mol, distance_matrix = input_mols[mol_name]

    result = top.diameter(mol, distance_matrix)
    result_no_dist_matrix = top.diameter(mol)

    assert result == expected_value
    assert result_no_dist_matrix == expected_value


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
        "sulfur": 0,
    }.items(),
)
def test_graph_distance_index(mol_name, expected_value, input_mols):
    mol, distance_matrix = input_mols[mol_name]

    result = top.graph_distance_index(mol, distance_matrix)
    result_no_dist_matrix = top.graph_distance_index(mol)

    assert result == expected_value
    assert result_no_dist_matrix == expected_value


@pytest.mark.parametrize(
    "mol_name, expected_value",
    {
        "ethane": 0.0,
        "ethanol": 1.0,
        "carbon_dioxide": 1.0,
        "benzene": 0.0,
        "acetic_acid": 1.0,
        "pyridine": 0.0,
        "isobutane": 1.0,
        "pyrimidine": 1.0,
        "sulfur": 0,
    }.items(),
)
def test_petitjean_index(mol_name, expected_value, input_mols):
    mol, distance_matrix = input_mols[mol_name]

    result = top.petitjean_index(mol)
    result_no_dist_matrix = top.petitjean_index(mol)

    assert round(result, 3) == expected_value
    assert result_no_dist_matrix == expected_value


@pytest.mark.parametrize(
    "mol_name, expected_value",
    {
        "ethane": 1,
        "ethanol": 1,
        "carbon_dioxide": 1,
        "benzene": 3,
        "acetic_acid": 1,
        "pyridine": 3,
        "isobutane": 1,
        "pyrimidine": 2,
        "sulfur": 0,
    }.items(),
)
def test_radius(mol_name, expected_value, input_mols):
    mol, distance_matrix = input_mols[mol_name]

    result = top.radius(mol, distance_matrix)
    result_no_dist_matrix = top.radius(mol)

    assert result == expected_value
    assert result_no_dist_matrix == expected_value


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
        "sulfur": 0,
    }.items(),
)
def test_zagreb_index_m1(mol_name, expected_value, input_mols):
    mol, _ = input_mols[mol_name]
    result = top.zagreb_index_m1(mol)
    assert result == expected_value


@pytest.mark.parametrize(
    "mol_name, expected_value",
    {
        "ethane": 1,
        "ethanol": 4,
        "carbon_dioxide": 4,
        "benzene": 24,
        "acetic_acid": 9,
        "pyridine": 24,
        "isobutane": 9,
        "pyrimidine": 49,
        "sulfur": 0,
    }.items(),
)
def test_zagreb_index_m2(mol_name, expected_value, input_mols):
    mol, _ = input_mols[mol_name]
    result = top.zagreb_index_m2(mol)
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
        "sulfur": 0,
    }.items(),
)
def test_polarity_number(mol_name, expected_value, input_mols):
    mol, distance_matrix = input_mols[mol_name]

    result = top.polarity_number(mol, distance_matrix)
    result_no_dist_matrix = top.polarity_number(mol)

    assert result == expected_value
    assert result_no_dist_matrix == expected_value


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
def test_polarity_number_carbon_only(mol_name, expected_value, input_mols):
    mol, distance_matrix = input_mols[mol_name]

    result = top.polarity_number(mol, distance_matrix, carbon_only=True)
    result_no_dist_matrix = top.polarity_number(mol, carbon_only=True)

    assert result == expected_value
    assert result_no_dist_matrix == expected_value


def test_polarity_number_no_carbon(input_mols):
    mol, distance_matrix = input_mols["dinitrogen"]
    with pytest.raises(
        ValueError,
        match="The molecule contains no carbon atoms.",
    ):
        top.polarity_number(mol, distance_matrix=distance_matrix, carbon_only=True)
