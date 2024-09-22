from typing import Union

import pytest
from rdkit import Chem
from rdkit.Chem import Mol

from skfp.model_selection.splitters.scaffold_split import (
    _create_scaffold_sets,
    scaffold_train_test_split,
    scaffold_train_valid_test_split,
)


@pytest.fixture
def all_molecules() -> list[str]:
    return [
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


@pytest.fixture
def smiles_ten_scaffolds() -> list[str]:
    ten_different_scaffolds = [
        "C1CCCC(C2CC2)CC1",
        "c1n[nH]cc1C1CCCCCC1",
        "c1n[nH]cc1CC1CCCCCC1",
        "C1CCCC(CC2CCOCC2)CC1",
        "c1ccc2nc(OC3CCC3)ccc2c1",
        "O=C(CCc1cscn1)NC1CCNCC1",
        "c1ccc2nc(OC3CCOC3)ccc2c1",
        "c1ccc2nc(NC3CCOCC3)ccc2c1",
        "c1ccc2nc(N3CCCOCC3)ccc2c1",
        "c1ccc2nc(N3CCn4ccnc4C3)ccc2c1",
    ]

    return ten_different_scaffolds


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
    scaffolds = _create_scaffold_sets(all_molecules)
    assert len(scaffolds) <= len(all_molecules)


def test_no_ring_molecules():
    smiles_list = ["CCO", "CCN", "CCC", "CCCl", "CCBr"]
    scaffolds = _create_scaffold_sets(smiles_list)
    assert len(scaffolds) == 1


def test_scaffold_count_for_benzodiazepines():
    smiles_list = [
        "C1CN=C(C2=CC=CC=C2)N=C1",
        "C1CN=C(C2=CC=CC=C2F)N=C1",
        "C1CN=C(C2=CC=CC=C2Cl)N=C1",
    ]

    scaffolds = _create_scaffold_sets(smiles_list)
    assert len(scaffolds) == 1


def test_scaffold_count_for_xanthines():
    smiles_list = [
        "O=c1[nH]c(=O)c2[nH]cnc2[nH]1",
        "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        "Cn1cnc2c1c(=O)[nH]c(=O)n2C",
        "Cn1c(=O)c2[nH]cnc2n(C)c1=O",
    ]
    scaffolds = _create_scaffold_sets(smiles_list)
    assert len(scaffolds) == 1


def test_csk_high_degree_atoms():
    smiles = ["O=[U]=O", "F[U](F)(F)(F)(F)F"]

    scaffold_train_test_split(smiles, use_csk=True)


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


def test_empty_train_subset_raises_an_error_train_test():
    smiles_list = [
        "C1CCCC(C2CC2)CC1",
    ]

    with pytest.raises(
        ValueError,
        match="Train subset is empty",
    ):
        scaffold_train_test_split(data=smiles_list)


def test_empty_train_subset_raises_an_error_train_valid_test():
    smiles_list = [
        "C1CCCC(C2CC2)CC1",
        "c1n[nH]cc1C1CCCCCC1",
    ]

    with pytest.raises(
        ValueError,
        match="Train subset is empty",
    ):
        scaffold_train_valid_test_split(data=smiles_list)


def test_train_test_split_properly_splits_csk_with_ints(
    smiles_ten_scaffolds,
):
    train_set, test_set = scaffold_train_test_split(
        data=smiles_ten_scaffolds, train_size=7, test_size=3, use_csk=True
    )
    assert len(train_set) == 7
    assert len(test_set) == 3


def test_train_test_split_properly_splits_csk_with_floats(
    smiles_ten_scaffolds,
):
    train_set, test_set = scaffold_train_test_split(
        data=smiles_ten_scaffolds,
        train_size=0.7,
        test_size=0.3,
        use_csk=True,
    )
    assert len(train_set) == 7
    assert len(test_set) == 3


def test_train_valid_test_split_properly_splits_csk_with_ints(
    smiles_ten_scaffolds,
):
    train_set, valid_set, test_set = scaffold_train_valid_test_split(
        data=smiles_ten_scaffolds,
        train_size=7,
        valid_size=2,
        test_size=1,
        use_csk=True,
    )
    assert len(train_set) == 7
    assert len(valid_set) == 2
    assert len(test_set) == 1


def test_train_valid_test_split_properly_splits_csk_with_floats(
    smiles_ten_scaffolds,
):
    train_set, valid_size, test_set = scaffold_train_valid_test_split(
        data=smiles_ten_scaffolds,
        train_size=0.7,
        valid_size=0.2,
        test_size=0.1,
        use_csk=True,
    )
    assert len(train_set) == 7
    assert len(valid_size) == 2
    assert len(test_set) == 1
