import pytest
from rdkit.Chem import Mol, MolFromSmiles

from skfp.distances import fraggle_distance, fraggle_similarity
from tests.distances.utils import assert_similarity_and_distance_values


def _get_values() -> list[tuple[Mol, Mol, str, float]]:
    paracetamol = MolFromSmiles("CC(=O)Nc1ccc(O)cc1")
    ibuprofen = MolFromSmiles("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O")
    caffeine = MolFromSmiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    theobromine = MolFromSmiles("Cn1cnc2c1c(=O)[nH]c(=O)n2C")

    return [
        (paracetamol, paracetamol, "==", 1.0),
        (ibuprofen, ibuprofen, "==", 1.0),
        (caffeine, caffeine, "==", 1.0),
        (theobromine, theobromine, "==", 1.0),
        (caffeine, theobromine, ">", 0.5),
        (paracetamol, ibuprofen, ">", 0.5),
        (ibuprofen, paracetamol, ">", 0.5),
        (paracetamol, caffeine, "<", 0.5),
        (caffeine, paracetamol, "<", 0.5),
    ]


@pytest.mark.parametrize("mol_query, mol_ref, comparison, value", _get_values())
def test_fraggle(mol_query, mol_ref, comparison, value):
    similarity = fraggle_similarity(mol_query, mol_ref)
    distance = fraggle_distance(mol_query, mol_ref)

    assert_similarity_and_distance_values(similarity, distance, comparison, value)
