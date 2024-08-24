import random
from typing import Union

import pytest
from numpy.random.mtrand import RandomState
from rdkit import Chem
from rdkit.Chem import Mol

from skfp.model_selection.randomized_scaffold_split import (
    _create_scaffolds,
    randomized_scaffold_train_test_split,
    randomized_scaffold_train_valid_test_split,
)


@pytest.fixture
def all_molecules() -> list[str]:
    all_smiles: list[str] = [
        "CCC",
        "CCCl",
        "CCBr",
        "C1CC1",
        "C1CCC1",
        "C1CCCC1",
        "C1CCCCC1",
        "CCO",
        "CCN",
        "CC.OCC",
    ]

    return all_smiles


@pytest.fixture
def additional_data() -> list[list[Union[str, int, bool]]]:
    return [
        ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [True, False, True, False, True, False, True, False, True, False],
    ]


def test_randomized_scaffold_creation_total_count(all_molecules):
    randomized_scaffolds = _create_scaffolds(all_molecules)
    assert len(randomized_scaffolds) <= len(all_molecules)


def test_no_ring_molecules():
    smiles_list: list[str] = ["CCO", "CCN", "CCC", "CCCl", "CCBr"]
    randomized_scaffolds = _create_scaffolds(smiles_list)
    assert len(randomized_scaffolds) == 1


def test_randomized_scaffold_count_for_benzodiazepines():
    smiles_list: list[str] = [
        "C1CN=C(C2=CC=CC=C2)N=C1",
        "C1CN=C(C2=CC=CC=C2F)N=C1",
        "C1CN=C(C2=CC=CC=C2Cl)N=C1",
    ]

    randomized_scaffolds = _create_scaffolds(smiles_list)
    assert len(randomized_scaffolds) == 1


def test_randomized_scaffold_count_for_xanthines():
    smiles = [
        "O=c1[nH]c(=O)c2[nH]cnc2[nH]1",
        "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        "Cn1cnc2c1c(=O)[nH]c(=O)n2C",
        "Cn1c(=O)c2[nH]cnc2n(C)c1=O",
    ]
    randomized_scaffolds = _create_scaffolds(smiles)
    assert len(randomized_scaffolds) == 1


def test_csk_should_fail_for_degree_greater_than_four():
    smiles = ["O=[U]=O", "F[U](F)(F)(F)(F)F"]

    with pytest.raises(
        Chem.rdchem.AtomValenceException,
        match="Explicit valence for atom # 1 C, 6, is greater than permitted",
    ):
        _ = randomized_scaffold_train_test_split(smiles, random_state=42, use_csk=True)


def test_randomized_scaffold_train_test_split_returns_molecules(all_molecules):
    mols = [Chem.MolFromSmiles(smiles) for smiles in all_molecules]
    train_set, test_set = randomized_scaffold_train_test_split(
        mols, random_state=42, return_indices=False
    )

    assert all(isinstance(train, Mol) for train in train_set)
    assert all(isinstance(test, Mol) for test in test_set)


def test_randomized_scaffold_train_test_split_return_indices(all_molecules):
    mols = [Chem.MolFromSmiles(smiles) for smiles in all_molecules]
    train_set, test_set = randomized_scaffold_train_test_split(
        mols, random_state=42, return_indices=False
    )

    assert all(isinstance(train, Mol) for train in train_set)
    assert all(isinstance(test, Mol) for test in test_set)


def test_randomized_scaffold_train_valid_test_split_returns_molecules(all_molecules):
    mols = [Chem.MolFromSmiles(smiles) for smiles in all_molecules]
    train_set, valid_set, test_set = randomized_scaffold_train_valid_test_split(
        mols, random_state=42, return_indices=False
    )

    assert all(isinstance(train, Mol) for train in train_set)
    assert all(isinstance(valid, Mol) for valid in valid_set)
    assert all(isinstance(test, Mol) for test in test_set)


def test_randomized_scaffold_train_valid_test_split_return_indices(all_molecules):
    mols = [Chem.MolFromSmiles(smiles) for smiles in all_molecules]
    train_idxs, valid_idxs, test_idxs = randomized_scaffold_train_valid_test_split(
        mols, random_state=42, return_indices=True
    )

    assert all(isinstance(idx, int) for idx in train_idxs)
    assert all(isinstance(idx, int) for idx in valid_idxs)
    assert all(isinstance(idx, int) for idx in test_idxs)


def test_seed_consistency_train_test_split(all_molecules):
    train_set_1, test_set_1 = randomized_scaffold_train_test_split(
        all_molecules, random_state=42
    )

    train_set_2, test_set_2 = randomized_scaffold_train_test_split(
        all_molecules, random_state=42
    )

    assert train_set_1 == train_set_2
    assert test_set_1 == test_set_2


def test_seed_consistency_train_valid__test_split(all_molecules):
    train_set_1, valid_set_1, test_set_1 = randomized_scaffold_train_valid_test_split(
        all_molecules, random_state=42
    )

    train_set_2, valid_set_2, test_set_2 = randomized_scaffold_train_valid_test_split(
        all_molecules, random_state=42
    )

    assert train_set_1 == train_set_2
    assert valid_set_1 == valid_set_2
    assert test_set_1 == test_set_2
