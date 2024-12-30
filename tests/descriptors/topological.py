import numpy as np
import pytest
from rdkit import Chem

from descriptors.topological import TopologicalDescriptors


@pytest.fixture
def diverse_molecules():
    return [
        ("ethane", Chem.MolFromSmiles("CC"), np.array([[0, 1], [1, 0]])),
        ("ethanol", Chem.MolFromSmiles("CCO"), None),
        (
            "carbon_dioxide",
            Chem.MolFromSmiles("O=C=O"),
            np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]]),
        ),
        (
            "benzene",
            Chem.MolFromSmiles("C1=CC=CC=C1"),
            np.array(
                [
                    [0, 1, 2, 3, 2, 1],
                    [1, 0, 1, 2, 3, 2],
                    [2, 1, 0, 1, 2, 3],
                    [2, 3, 2, 1, 0, 1],
                    [2, 1, 2, 1, 0, 1],
                    [1, 2, 3, 2, 1, 0],
                ]
            ),
        ),
        (
            "acetic_acid",
            Chem.MolFromSmiles("CC(=O)O"),
            np.array([[0, 1, 2, 2], [1, 0, 1, 1], [2, 1, 0, 2], [2, 1, 2, 0]]),
        ),
        ("pyridine", Chem.MolFromSmiles("c1ccncc1"), None),
        ("isobutane", Chem.MolFromSmiles("C(C)(C)C"), None),
        ("pyrimidine", Chem.MolFromSmiles("c1cnc2ncnc12"), None),
    ]


@pytest.mark.parametrize(
    "mol_name, descriptor, expected_value",
    [
        ("ethane", "weiner_index", 1.0),
        ("ethanol", "weiner_index", 4.0),
        ("carbon_dioxide", "average_weiner_index", 1.3333333333333333),
        ("acetic_acid", "average_weiner_index", 1.5),
        ("benzene", "zagreb_index", 24.0),
        ("isobutane", "zagreb_index", 12),
        ("pyridine", "graph_distance_index", 261.0),
        ("pyrimidine", "graph_distance_index", 997.0),
    ],
)
def test_descriptors(diverse_molecules, mol_name, descriptor, expected_value):
    mol_tuple = next(item for item in diverse_molecules if item[0] == mol_name)
    mol, distance_matrix = mol_tuple[1], mol_tuple[2]
    topological_descriptor = TopologicalDescriptors(mol, distance_matrix)

    result = getattr(topological_descriptor, descriptor)()

    assert result == pytest.approx(expected_value, abs=1e-6)
