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
    ]:
        mol = MolFromSmiles(smiles)
        dist_matrix = GetDistanceMatrix(mol)
        result[name] = (mol, dist_matrix)

    return result


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
def test_wiener_index(mol_name, expected_value, input_mols):
    mol, distance_matrix = input_mols[mol_name]

    result = top.wiener_index(mol, distance_matrix)
    result_no_dist_matrix = top.wiener_index(mol)

    assert result == expected_value
    assert result_no_dist_matrix == expected_value


@pytest.mark.parametrize(
    "mol_name, expected_value",
    [
        ("ethane", 1.0),
        ("ethanol", 1.333),
        ("carbon_dioxide", 1.333),
        ("benzene", 1.8),
        ("acetic_acid", 1.5),
        ("pyridine", 1.8),
        ("isobutane", 1.5),
        ("pyrimidine", 1.964),
    ],
)
def test_average_wiener_index(mol_name, expected_value, input_mols):
    mol, distance_matrix = input_mols[mol_name]

    result = top.average_wiener_index(mol, distance_matrix)
    result_no_dist_matrix = top.average_wiener_index(mol)

    assert round(result, 3) == expected_value
    assert round(result_no_dist_matrix, 3) == expected_value


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
def test_graph_distance_index(mol_name, expected_value, input_mols):
    mol, distance_matrix = input_mols[mol_name]

    result = top.graph_distance_index(mol, distance_matrix)
    result_no_dist_matrix = top.graph_distance_index(mol)

    assert result == expected_value
    assert result_no_dist_matrix == expected_value


@pytest.mark.parametrize(
    "mol_name, expected_value",
    {
        "ethane": 2,
        "ethanol": 2,
        "carbon_dioxide": 2,
        "benzene": 1.0,
        "acetic_acid": 2,
        "pyridine": 1.0,
        "isobutane": 2.0,
        "pyrimidine": 0,
    }.items(),
)
def test_hall_kier_alpha(mol_name, expected_value, input_mols):
    mol, _ = input_mols[mol_name]
    result = top.hall_kier_alpha(mol)
    assert result == expected_value


@pytest.mark.parametrize(
    "mol_name, expected_value",
    {
        "ethane": 36.0,
        "ethanol": 20.0,
        "carbon_dioxide": 0.0,
        "benzene": 0.0,
        "acetic_acid": 37.5,
        "pyridine": 0.0,
        "isobutane": 16.666667,
        "pyrimidine": 15.68,
    }.items(),
)
def test_kappa1_index(mol_name, expected_value, input_mols):
    mol, _ = input_mols[mol_name]
    result = top.kappa1_index(mol)
    assert round(result, 6) == round(expected_value, 6)


@pytest.mark.parametrize(
    "mol_name, expected_value",
    {
        "ethane": 0.0,
        "ethanol": 36.0,
        "carbon_dioxide": 36.0,
        "benzene": 4.167,
        "acetic_acid": 8.889,
        "pyridine": 4.167,
        "isobutane": 8.889,
        "pyrimidine": 1.75,
    }.items(),
)
def test_kappa2_index(mol_name, expected_value, input_mols):
    mol, distance_matrix = input_mols[mol_name]
    result = top.kappa2_index(mol, distance_matrix)
    result_no_dist_matrix = top.kappa2_index(mol)

    assert round(result, 3) == round(expected_value, 3)
    assert round(result_no_dist_matrix, 3) == round(expected_value, 3)


@pytest.mark.parametrize(
    "mol_name, expected_value",
    {
        "ethane": 0.0,
        "ethanol": 0.0,
        "carbon_dioxide": 0.0,
        "benzene": 10.666667,
        "acetic_acid": 0.0,
        "pyridine": 10.666667,
        "isobutane": 0.0,
        "pyrimidine": 4.861111,
    }.items(),
)
def test_kappa3_index(mol_name, expected_value, input_mols):
    mol, distance_matrix = input_mols[mol_name]
    result = top.kappa3_index(mol, distance_matrix)
    result_no_dist_matrix = top.kappa3_index(mol)

    assert round(result, 6) == round(expected_value, 6)
    assert round(result_no_dist_matrix, 6) == round(expected_value, 6)


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
def test_zagreb_index(mol_name, expected_value, input_mols):
    mol, _ = input_mols[mol_name]
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
        match="The molecule contains no carbon atoms",
    ):
        top.polarity_number(mol, distance_matrix=distance_matrix, carbon_only=True)
