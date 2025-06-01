import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import Mol

from skfp.model_selection.splitters.maxmin_split import (
    maxmin_stratified_train_test_split,
    maxmin_stratified_train_valid_test_split,
    maxmin_train_test_split,
    maxmin_train_valid_test_split,
)


@pytest.fixture
def varied_mols() -> list[str]:
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


def test_maxmin_split_size(varied_mols):
    train_set, test_set = maxmin_train_test_split(
        varied_mols, train_size=0.7, test_size=0.3
    )

    assert len(train_set) == 7
    assert len(test_set) == 3

    train_set, test_set = maxmin_train_test_split(
        varied_mols, train_size=0.6, test_size=0.4
    )

    assert len(train_set) == 6
    assert len(test_set) == 4


def test_maxmin_valid_split_size(varied_mols):
    train_set, valid_set, test_set = maxmin_train_valid_test_split(
        varied_mols, train_size=0.7, test_size=0.2, valid_size=0.1
    )

    assert len(train_set) == 7
    assert len(test_set) == 2
    assert len(valid_set) == 1

    train_set, valid_set, test_set = maxmin_train_valid_test_split(
        varied_mols, train_size=0.5, test_size=0.2, valid_size=0.3
    )

    assert len(train_set) == 5
    assert len(test_set) == 2
    assert len(valid_set) == 3


def test_seed_consistency_train_valid_test_split(varied_mols):
    train_set_1, valid_set_1, test_set_1 = maxmin_train_valid_test_split(
        varied_mols, train_size=0.7, valid_size=0.1, test_size=0.2, random_state=0
    )

    train_set_2, valid_set_2, test_set_2 = maxmin_train_valid_test_split(
        varied_mols, train_size=0.7, valid_size=0.1, test_size=0.2, random_state=0
    )

    assert train_set_1 == train_set_2
    assert valid_set_1 == valid_set_2
    assert test_set_1 == test_set_2


def test_seed_consistency_train_test_split(varied_mols):
    train_set_1, test_set_1 = maxmin_train_test_split(
        varied_mols, train_size=0.7, test_size=0.3, random_state=0
    )

    train_set_2, test_set_2 = maxmin_train_test_split(
        varied_mols, train_size=0.7, test_size=0.3, random_state=0
    )

    assert train_set_1 == train_set_2
    assert test_set_1 == test_set_2


def test_maxmin_train_test_split_return_molecules(varied_mols):
    mols = [Chem.MolFromSmiles(smiles) for smiles in varied_mols]
    train_set, test_set = maxmin_train_test_split(mols, train_size=0.7, test_size=0.3)

    assert all(isinstance(train, Mol) for train in train_set)
    assert all(isinstance(test, Mol) for test in test_set)

    assert len(train_set + test_set) == len(mols)


def test_maxmin_train_valid_test_split_returns_molecules(varied_mols):
    mols = [Chem.MolFromSmiles(smiles) for smiles in varied_mols]
    train_set, valid_set, test_set = maxmin_train_valid_test_split(
        mols,
        train_size=0.7,
        test_size=0.2,
        valid_size=0.1,
        return_indices=False,
    )

    assert all(isinstance(train, Mol) for train in train_set)
    assert all(isinstance(valid, Mol) for valid in valid_set)
    assert all(isinstance(test, Mol) for test in test_set)

    assert len(train_set + valid_set + test_set) == len(mols)


def test_maxmin_train_test_split_return_indices(varied_mols):
    mols = [Chem.MolFromSmiles(smiles) for smiles in varied_mols]
    train_set, test_set = maxmin_train_test_split(
        mols, train_size=0.7, test_size=0.3, return_indices=True
    )

    assert all(isinstance(train, int) for train in train_set)
    assert all(isinstance(test, int) for test in test_set)

    assert len(set(train_set) | set(test_set)) == len(varied_mols)


def test_maxmin_train_valid_test_split_returns_indices(varied_mols):
    mols = [Chem.MolFromSmiles(smiles) for smiles in varied_mols]
    train_set, valid_set, test_set = maxmin_train_valid_test_split(
        mols,
        train_size=0.7,
        test_size=0.2,
        valid_size=0.1,
        return_indices=True,
    )

    assert all(isinstance(train, int) for train in train_set)
    assert all(isinstance(valid, int) for valid in valid_set)
    assert all(isinstance(test, int) for test in test_set)

    assert len(set(train_set) | set(valid_set) | set(test_set)) == len(varied_mols)


def test_maxmin_train_test_split_with_additional_data(varied_mols):
    mols = [Chem.MolFromSmiles(smiles) for smiles in varied_mols]
    labels = np.ones(len(varied_mols))
    train_mols, test_mols, train_labels, test_labels = maxmin_train_test_split(
        mols, labels
    )

    assert len(train_mols) == len(train_labels)
    assert len(test_mols) == len(test_labels)


def test_maxmin_train_valid_test_split_with_additional_data(varied_mols):
    mols = [Chem.MolFromSmiles(smiles) for smiles in varied_mols]
    labels = np.ones(len(varied_mols))
    train_mols, valid_mols, test_mols, train_labels, valid_labels, test_labels = (
        maxmin_train_valid_test_split(mols, labels)
    )

    assert len(train_mols) == len(train_labels)
    assert len(valid_mols) == len(valid_labels)
    assert len(test_mols) == len(test_labels)


def test_maxmin_stratified_train_test_split(mols_list):
    labels = [0] * int(0.5 * len(mols_list)) + [1] * int(0.5 * len(mols_list))

    mols_train, mols_test, y_train, y_test = maxmin_stratified_train_test_split(
        mols_list, labels, train_size=0.7, test_size=0.3
    )

    assert len(mols_train) == int(0.7 * len(mols_list))
    assert len(mols_test) == int(0.3 * len(mols_list))
    assert len(mols_train + mols_test) == len(mols_list)

    assert len(mols_train) == len(y_train)
    assert len(mols_test) == len(y_test)

    assert np.isclose(y_train.mean(), 0.5)
    assert np.isclose(y_test.mean(), 0.5)
    assert len(np.concatenate((y_train, y_test))) == len(mols_list)


def test_maxmin_stratified_train_valid_test_split(mols_list):
    labels = [0] * int(0.5 * len(mols_list)) + [1] * int(0.5 * len(mols_list))

    mols_train, mols_valid, mols_test, y_train, y_valid, y_test = (
        maxmin_stratified_train_valid_test_split(
            mols_list,
            labels,
            train_size=0.7,
            valid_size=0.1,
            test_size=0.2,
        )
    )

    assert len(mols_train) == int(0.7 * len(mols_list))
    assert len(mols_valid) == int(0.1 * len(mols_list))
    assert len(mols_test) == int(0.2 * len(mols_list))
    assert len(mols_train + mols_valid + mols_test) == len(mols_list)

    assert len(mols_train) == len(y_train)
    assert len(mols_valid) == len(y_valid)
    assert len(mols_test) == len(y_test)

    assert np.isclose(y_train.mean(), 0.5)
    assert np.isclose(y_valid.mean(), 0.5)
    assert np.isclose(y_test.mean(), 0.5)
    assert len(np.concatenate((y_train, y_valid, y_test))) == len(mols_list)


def test_maxmin_stratified_train_test_split_return_indices(mols_list):
    labels = [0] * int(0.5 * len(mols_list)) + [1] * int(0.5 * len(mols_list))

    train_set, test_set, train_labels, test_labels = maxmin_stratified_train_test_split(
        mols_list, labels, train_size=0.7, test_size=0.3, return_indices=True
    )

    assert all(isinstance(train, int) for train in train_set)
    assert all(isinstance(test, int) for test in test_set)

    assert len(set(train_set) | set(test_set)) == len(mols_list)


def test_maxmin_stratified_train_valid_test_split_returns_indices(mols_list):
    labels = [0] * int(0.5 * len(mols_list)) + [1] * int(0.5 * len(mols_list))

    train_set, valid_set, test_set, train_labels, valid_labels, test_labels = (
        maxmin_stratified_train_valid_test_split(
            mols_list,
            labels,
            train_size=0.7,
            test_size=0.2,
            valid_size=0.1,
            return_indices=True,
        )
    )

    assert all(isinstance(train, int) for train in train_set)
    assert all(isinstance(valid, int) for valid in valid_set)
    assert all(isinstance(test, int) for test in test_set)

    assert len(set(train_set) | set(valid_set) | set(test_set)) == len(mols_list)


def test_maxmin_stratified_train_test_split_with_additional_data(mols_list):
    labels = [0] * int(0.5 * len(mols_list)) + [1] * int(0.5 * len(mols_list))
    additional_data = ["a"] * len(mols_list)

    (
        train_mols,
        test_mols,
        train_labels,
        test_labels,
        train_additional_data,
        test_additional_data,
    ) = maxmin_stratified_train_test_split(
        mols_list, labels, additional_data, test_size=0.2
    )

    assert len(train_mols) == len(train_labels) == len(train_additional_data)
    assert len(test_mols) == len(test_labels) == len(test_additional_data)


def test_maxmin_stratified_train_valid_test_split_with_additional_data(mols_list):
    labels = [0] * int(0.5 * len(mols_list)) + [1] * int(0.5 * len(mols_list))
    additional_data = ["a"] * len(mols_list)

    (
        train_mols,
        valid_mols,
        test_mols,
        train_labels,
        valid_labels,
        test_labels,
        train_additional_data,
        valid_additional_data,
        test_additional_data,
    ) = maxmin_stratified_train_valid_test_split(
        mols_list,
        labels,
        additional_data,
        train_size=0.7,
        valid_size=0.1,
        test_size=0.2,
    )

    assert len(train_mols) == len(train_labels) == len(train_additional_data)
    assert len(valid_mols) == len(valid_labels) == len(valid_additional_data)
    assert len(test_mols) == len(test_labels) == len(test_additional_data)
