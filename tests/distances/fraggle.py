import pytest
from numpy.testing import assert_allclose
from rdkit.Chem import Mol, MolFromSmiles

from skfp.distances import (
    bulk_fraggle_distance,
    bulk_fraggle_similarity,
    fraggle_distance,
    fraggle_similarity,
)


def _get_values() -> list[tuple[Mol, Mol, float, float]]:
    # mol_query, mol_ref, similarity, distance
    paracetamol = MolFromSmiles("CC(=O)Nc1ccc(O)cc1")
    ibuprofen = MolFromSmiles("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O")
    caffeine = MolFromSmiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
    theobromine = MolFromSmiles("Cn1cnc2c1c(=O)[nH]c(=O)n2C")

    return [
        (paracetamol, paracetamol, 1.0, 0.0),
        (ibuprofen, ibuprofen, 1.0, 0.0),
        (caffeine, caffeine, 1.0, 0.0),
        (theobromine, theobromine, 1.0, 0.0),
        (caffeine, theobromine, 0.5, 0.5),
        (paracetamol, ibuprofen, 0.5, 0.5),
        (ibuprofen, paracetamol, 0.5, 0.5),
        (paracetamol, caffeine, 0.1, 0.9),
        (caffeine, paracetamol, 0.1, 0.9),
    ]


@pytest.mark.parametrize(
    "mol_query, mol_ref, similarity_threshold, distance_threshold", _get_values()
)
def test_fraggle(mol_query, mol_ref, similarity_threshold, distance_threshold):
    computed_similarity = fraggle_similarity(mol_query, mol_ref)
    computed_distance = fraggle_distance(mol_query, mol_ref)

    assert computed_similarity >= similarity_threshold
    assert computed_distance <= distance_threshold


def test_bulk_fraggle(mols_list):
    mols_list = mols_list[:5]

    pairwise_sim = [
        [fraggle_similarity(mols_list[i], mols_list[j]) for j in range(len(mols_list))]
        for i in range(len(mols_list))
    ]
    pairwise_dist = [
        [fraggle_distance(mols_list[i], mols_list[j]) for j in range(len(mols_list))]
        for i in range(len(mols_list))
    ]

    bulk_sim = bulk_fraggle_similarity(mols_list)
    bulk_dist = bulk_fraggle_distance(mols_list)

    assert_allclose(pairwise_sim, bulk_sim, atol=1e-3)
    assert_allclose(pairwise_dist, bulk_dist, atol=1e-3)


def test_bulk_fraggle_second_list(mols_list):
    mols_list = mols_list[:5]

    bulk_sim_single = bulk_fraggle_similarity(mols_list)
    bulk_sim_two = bulk_fraggle_similarity(mols_list, mols_list)

    assert_allclose(bulk_sim_single, bulk_sim_two, atol=1e-3)
