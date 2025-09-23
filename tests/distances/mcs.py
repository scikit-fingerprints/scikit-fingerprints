import numpy as np
import pytest
from rdkit.Chem import Mol, MolFromSmiles

from skfp.distances import (
    bulk_mcs_distance,
    bulk_mcs_similarity,
    mcs_distance,
    mcs_similarity,
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
        (paracetamol, caffeine, 0.5, 0.5),
        (caffeine, paracetamol, 0.5, 0.5),
    ]


@pytest.mark.parametrize("mol_query, mol_ref, similarity, distance", _get_values())
def test_mcs(mol_query, mol_ref, comparison, similarity, distance):
    computed_similarity = mcs_similarity(mol_query, mol_ref)
    computed_distance = mcs_distance(mol_query, mol_ref)

    assert np.isclose(computed_similarity, similarity)
    assert np.isclose(computed_distance, distance)


def test_bulk_mcs(mols_list):
    mols_list = mols_list[:5]

    pairwise_sim = [
        [mcs_similarity(mols_list[i], mols_list[j]) for j in range(len(mols_list))]
        for i in range(len(mols_list))
    ]
    pairwise_dist = [
        [mcs_distance(mols_list[i], mols_list[j]) for j in range(len(mols_list))]
        for i in range(len(mols_list))
    ]

    bulk_sim = bulk_mcs_similarity(mols_list)
    bulk_dist = bulk_mcs_distance(mols_list)

    assert np.allclose(pairwise_sim, bulk_sim)
    assert np.allclose(pairwise_dist, bulk_dist)


def test_bulk_mcs_second_list(mols_list):
    mols_list = mols_list[:5]

    bulk_sim_single = bulk_mcs_similarity(mols_list)
    bulk_sim_two = bulk_mcs_similarity(mols_list, mols_list)
    assert np.allclose(bulk_sim_single, bulk_sim_two)
