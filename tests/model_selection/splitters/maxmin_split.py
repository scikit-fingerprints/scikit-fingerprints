from typing import Union

import pytest
from rdkit import Chem
from rdkit.Chem import Mol

from skfp.model_selection.splitters.maxmin_split import (
    maxmin_train_test_split,
    maxmin_train_valid_test_split,
)


@pytest.fixture
def all_molecules() -> list[str]:
    return [
        "OCC3OC(OCC2OC(OC(C#N)c1ccccc1)C(O)C(O)C2O)C(O)C(O)C3O",
        "Cc1occc1C(=O)Nc2ccccc2",
        "CCCCCCCCCCC",
        "c1ccc2c(c1)ccc3c2ccc4c5ccccc5ccc43",
        "CCCCCCCCCCCC",
        "CCCCCCCCCCCCC",
        "Clc1cc(Cl)c(c(Cl)c1)c2c(Cl)cccc2Cl",
        "C1CCNCC1",
        "ClC4=C(Cl)C5(Cl)C3C1CC(C2OC12)C3C4(Cl)C5(Cl)Cl",
        "COc5cc4OCC3Oc2c1CC(Oc1ccc2C(=O)C3c4cc5OC)C(C)=C",
    ]


@pytest.fixture
def additional_data() -> list[list[Union[str, int, bool]]]:
    return [
        ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [True, False, True, False, True, False, True, False, True, False],
    ]


def test_maxmin_split_size(all_molecules):

    train_set, test_set = maxmin_train_test_split(
        all_molecules, train_size=0.7, test_size=0.3, random_state=0
    )

    assert len(train_set) == 7
    assert len(test_set) == 3

    train_set, test_set = maxmin_train_test_split(
        all_molecules, train_size=0.6, test_size=0.4, random_state=0
    )

    assert len(train_set) == 6
    assert len(test_set) == 4


def test_maxmin_valid_split_size(all_molecules):

    train_set, valid_set, test_set = maxmin_train_valid_test_split(
        all_molecules, train_size=0.7, test_size=0.2, valid_size=0.1, random_state=0
    )

    assert len(train_set) == 7
    assert len(test_set) == 2
    assert len(valid_set) == 1

    train_set, valid_set, test_set = maxmin_train_valid_test_split(
        all_molecules, train_size=0.5, test_size=0.2, valid_size=0.3, random_state=0
    )

    assert len(train_set) == 5
    assert len(test_set) == 2
    assert len(valid_set) == 3


def test_seed_consistency_train_valid_test_split(all_molecules):

    train_set_1, valid_set_1, test_set_1 = maxmin_train_valid_test_split(
        all_molecules, train_size=0.7, valid_size=0.1, test_size=0.2, random_state=0
    )

    train_set_2, valid_set_2, test_set_2 = maxmin_train_valid_test_split(
        all_molecules, train_size=0.7, valid_size=0.1, test_size=0.2, random_state=0
    )

    assert train_set_1 == train_set_2
    assert valid_set_1 == valid_set_2
    assert test_set_1 == test_set_2


def test_seed_consistency_train_test_split(all_molecules):

    train_set_1, test_set_1 = maxmin_train_test_split(
        all_molecules, train_size=0.7, test_size=0.3, random_state=0
    )

    train_set_2, test_set_2 = maxmin_train_test_split(
        all_molecules, train_size=0.7, test_size=0.3, random_state=0
    )

    assert train_set_1 == train_set_2
    assert test_set_1 == test_set_2


def test_seed_diversity_train_valid_test_split(all_molecules):

    train_set_1, valid_set_1, test_set_1 = maxmin_train_valid_test_split(
        all_molecules, train_size=0.7, valid_size=0.1, test_size=0.2, random_state=0
    )

    train_set_2, valid_set_2, test_set_2 = maxmin_train_valid_test_split(
        all_molecules, train_size=0.7, valid_size=0.1, test_size=0.2, random_state=1
    )

    assert (
        train_set_1 != train_set_2
        or valid_set_1 != valid_set_2
        or test_set_1 != test_set_2
    )


def test_seed_diversity_train_test_split(all_molecules):

    train_set_1, test_set_1 = maxmin_train_test_split(
        all_molecules, train_size=0.7, test_size=0.3, random_state=0
    )

    train_set_2, test_set_2 = maxmin_train_test_split(
        all_molecules, train_size=0.7, test_size=0.3, random_state=1
    )

    assert train_set_1 != train_set_2 or test_set_1 != test_set_2


def test_maxmin_train_test_split_return_molecules(all_molecules):
    mols = [Chem.MolFromSmiles(smiles) for smiles in all_molecules]
    train_set, test_set = maxmin_train_test_split(
        mols, train_size=0.7, test_size=0.3, random_state=1, return_indices=False
    )

    assert all(isinstance(train, Mol) for train in train_set)
    assert all(isinstance(test, Mol) for test in test_set)


def test_maxmin_train_valid_test_split_returns_molecules(all_molecules):
    mols = [Chem.MolFromSmiles(smiles) for smiles in all_molecules]
    train_set, valid_set, test_set = maxmin_train_valid_test_split(
        mols,
        train_size=0.7,
        test_size=0.2,
        valid_size=0.1,
        random_state=1,
        return_indices=False,
    )

    assert all(isinstance(train, Mol) for train in train_set)
    assert all(isinstance(valid, Mol) for valid in valid_set)
    assert all(isinstance(test, Mol) for test in test_set)


def test_maxmin_train_test_split_return_indices(all_molecules):
    mols = [Chem.MolFromSmiles(smiles) for smiles in all_molecules]
    train_set, test_set = maxmin_train_test_split(
        mols, train_size=0.7, test_size=0.3, random_state=1, return_indices=True
    )

    assert all(isinstance(train, int) for train in train_set)
    assert all(isinstance(test, int) for test in test_set)


def test_maxmin_train_valid_test_split_returns_indices(all_molecules):
    mols = [Chem.MolFromSmiles(smiles) for smiles in all_molecules]
    train_set, valid_set, test_set = maxmin_train_valid_test_split(
        mols,
        train_size=0.7,
        test_size=0.2,
        valid_size=0.1,
        random_state=1,
        return_indices=True,
    )

    assert all(isinstance(train, int) for train in train_set)
    assert all(isinstance(valid, int) for valid in valid_set)
    assert all(isinstance(test, int) for test in test_set)
