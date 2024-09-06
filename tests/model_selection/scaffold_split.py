from typing import Union

import pytest
from rdkit import Chem
from rdkit.Chem import Mol

from skfp.model_selection.scaffold_split import (
    _create_scaffolds,
    scaffold_train_test_split,
    scaffold_train_valid_test_split,
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


def test_scaffold_train_test_split_default(all_molecules):
    train, test = scaffold_train_test_split(all_molecules)
    assert len(train) == 8
    assert len(test) == 2


def test_scaffold_train_test_split_custom_sizes(all_molecules):
    train, test = scaffold_train_test_split(
        all_molecules, train_size=0.7, test_size=0.3
    )
    assert len(train) == 7
    assert len(test) == 3


def test_train_test_split_total_molecule_count(all_molecules):
    train_split, test_split = scaffold_train_test_split(
        all_molecules, train_size=0.8, test_size=0.2
    )
    assert len(train_split) + len(test_split) == len(all_molecules)
    assert len(train_split) == 8
    assert len(test_split) == 2


def test_scaffold_train_test_split_with_additional_data(all_molecules, additional_data):
    train_set, test_set, additional = scaffold_train_test_split(
        all_molecules, *additional_data, train_size=0.7, test_size=0.3
    )

    train_additional = additional[0]
    test_additional = additional[1]

    assert len(train_set) == 7
    assert len(test_set) == 3
    assert len(train_additional) == 7
    assert len(test_additional) == 3


def test_scaffold_train_valid_test_split_with_additional_data(
    all_molecules, additional_data
):
    train_set, valid_set, test_set, additional = scaffold_train_valid_test_split(
        all_molecules, *additional_data, train_size=0.7, valid_size=0.2, test_size=0.1
    )

    train_additional = additional[0]
    valid_additional = additional[1]
    test_additional = additional[2]

    assert len(train_set) == 7
    assert len(valid_set) == 2
    assert len(test_set) == 1
    assert len(train_additional) == 7
    assert len(valid_additional) == 2
    assert len(test_additional) == 1


def test_train_valid_test_split_total_molecule_count(all_molecules):
    train_split, valid_split, test_split = scaffold_train_valid_test_split(
        all_molecules, train_size=0.8, valid_size=0.1, test_size=0.1
    )

    assert len(train_split) + len(valid_split) + len(test_split) == len(all_molecules)
    assert len(train_split) == 8
    assert len(valid_split) == 1
    assert len(test_split) == 1


def test_test_split_smaller_than_train_split(all_molecules):
    train_split, test_split = scaffold_train_test_split(
        all_molecules, train_size=0.7, test_size=0.3
    )

    assert len(train_split) > len(test_split)
    assert len(train_split) == 7
    assert len(test_split) == 3


def test_train_split_larger_than_valid_and_test_splits(all_molecules):
    train_split, valid_split, test_split = scaffold_train_valid_test_split(
        all_molecules, train_size=0.7, valid_size=0.2, test_size=0.1
    )

    assert len(train_split) > len(valid_split)
    assert len(valid_split) > len(test_split)
    assert len(train_split) == 7
    assert len(valid_split) == 2
    assert len(test_split) == 1


def test_scaffold_creation_total_count(all_molecules):
    scaffolds = _create_scaffolds(all_molecules)
    assert len(scaffolds) <= len(all_molecules)


def test_no_ring_molecules():
    smiles_list = ["CCO", "CCN", "CCC", "CCCl", "CCBr"]
    scaffolds = _create_scaffolds(smiles_list)
    assert len(scaffolds) == 1


def test_scaffold_count_for_benzodiazepines():
    smiles_list = [
        "C1CN=C(C2=CC=CC=C2)N=C1",
        "C1CN=C(C2=CC=CC=C2F)N=C1",
        "C1CN=C(C2=CC=CC=C2Cl)N=C1",
    ]

    scaffolds = _create_scaffolds(smiles_list)
    assert len(scaffolds) == 1


def test_scaffold_count_for_xanthines():
    smiles_list = [
        "O=c1[nH]c(=O)c2[nH]cnc2[nH]1",
        "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        "Cn1cnc2c1c(=O)[nH]c(=O)n2C",
        "Cn1c(=O)c2[nH]cnc2n(C)c1=O",
    ]
    scaffolds = _create_scaffolds(smiles_list)
    assert len(scaffolds) == 1


def test_csk_should_fail_for_degree_greater_than_four():
    smiles = ["O=[U]=O", "F[U](F)(F)(F)(F)F"]

    with pytest.raises(
        Chem.rdchem.AtomValenceException,
        match="Explicit valence for atom # 1 C, 6, is greater than permitted",
    ):
        _ = scaffold_train_test_split(smiles, use_csk=True)


def test_scaffold_train_test_split_returns_molecules(all_molecules):
    mols = [Chem.MolFromSmiles(smiles) for smiles in all_molecules]
    train_set, test_set = scaffold_train_test_split(mols, return_indices=False)

    assert all(isinstance(train, Mol) for train in train_set)
    assert all(isinstance(test, Mol) for test in test_set)


def test_scaffold_train_test_split_return_indices(all_molecules):
    mols = [Chem.MolFromSmiles(smiles) for smiles in all_molecules]
    train_idxs, test_idxs = scaffold_train_test_split(mols, return_indices=True)

    assert all(isinstance(idx, int) for idx in train_idxs)
    assert all(isinstance(idx, int) for idx in test_idxs)


def test_scaffold_train_valid_test_split_returns_molecules(all_molecules):
    mols = [Chem.MolFromSmiles(smiles) for smiles in all_molecules]
    train_set, valid_set, test_set = scaffold_train_valid_test_split(
        mols, return_indices=False
    )

    assert all(isinstance(train, Mol) for train in train_set)
    assert all(isinstance(valid, Mol) for valid in valid_set)
    assert all(isinstance(test, Mol) for test in test_set)


def test_scaffold_train_valid_test_split_return_indices(all_molecules):
    mols = [Chem.MolFromSmiles(smiles) for smiles in all_molecules]
    train_idxs, valid_idxs, test_idxs = scaffold_train_valid_test_split(
        mols, return_indices=True
    )

    assert all(isinstance(idx, int) for idx in train_idxs)
    assert all(isinstance(idx, int) for idx in valid_idxs)
    assert all(isinstance(idx, int) for idx in test_idxs)
