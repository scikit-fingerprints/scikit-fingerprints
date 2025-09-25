import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
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
    train, test = maxmin_train_test_split(varied_mols, train_size=0.7, test_size=0.3)
    assert_equal(len(train), 7)
    assert_equal(len(test), 3)

    train, test = maxmin_train_test_split(varied_mols, train_size=0.6, test_size=0.4)
    assert_equal(len(train), 6)
    assert_equal(len(test), 4)


def test_maxmin_valid_split_size(varied_mols):
    train, valid, test = maxmin_train_valid_test_split(
        varied_mols, train_size=0.7, test_size=0.2, valid_size=0.1
    )
    assert_equal(len(train), 7)
    assert_equal(len(test), 2)
    assert_equal(len(valid), 1)

    train, valid, test = maxmin_train_valid_test_split(
        varied_mols, train_size=0.5, test_size=0.2, valid_size=0.3
    )
    assert_equal(len(train), 5)
    assert_equal(len(test), 2)
    assert_equal(len(valid), 3)


def test_seed_consistency_train_valid_test_split(varied_mols):
    t1, v1, s1 = maxmin_train_valid_test_split(
        varied_mols, train_size=0.7, valid_size=0.1, test_size=0.2, random_state=0
    )
    t2, v2, s2 = maxmin_train_valid_test_split(
        varied_mols, train_size=0.7, valid_size=0.1, test_size=0.2, random_state=0
    )
    assert_equal(t1, t2)
    assert_equal(v1, v2)
    assert_equal(s1, s2)


def test_seed_consistency_train_test_split(varied_mols):
    t1, s1 = maxmin_train_test_split(
        varied_mols, train_size=0.7, test_size=0.3, random_state=0
    )
    t2, s2 = maxmin_train_test_split(
        varied_mols, train_size=0.7, test_size=0.3, random_state=0
    )
    assert_equal(t1, t2)
    assert_equal(s1, s2)


def test_maxmin_train_test_split_return_molecules(varied_mols):
    mols = [Chem.MolFromSmiles(smi) for smi in varied_mols]
    train, test = maxmin_train_test_split(mols, train_size=0.7, test_size=0.3)

    assert all(isinstance(m, Mol) for m in train)
    assert all(isinstance(m, Mol) for m in test)
    assert_equal(len(train) + len(test), len(mols))


def test_maxmin_train_valid_test_split_returns_molecules(varied_mols):
    mols = [Chem.MolFromSmiles(smi) for smi in varied_mols]
    train, valid, test = maxmin_train_valid_test_split(
        mols, train_size=0.7, test_size=0.2, valid_size=0.1, return_indices=False
    )

    assert all(isinstance(m, Mol) for m in train)
    assert all(isinstance(m, Mol) for m in valid)
    assert all(isinstance(m, Mol) for m in test)
    assert_equal(len(train) + len(valid) + len(test), len(mols))


def test_maxmin_train_test_split_return_indices(varied_mols):
    mols = [Chem.MolFromSmiles(smi) for smi in varied_mols]
    train, test = maxmin_train_test_split(
        mols, train_size=0.7, test_size=0.3, return_indices=True
    )

    assert all(isinstance(idx, int) for idx in train)
    assert all(isinstance(idx, int) for idx in test)
    assert_equal(len(set(train) | set(test)), len(varied_mols))


def test_maxmin_train_valid_test_split_returns_indices(varied_mols):
    mols = [Chem.MolFromSmiles(smi) for smi in varied_mols]
    train, valid, test = maxmin_train_valid_test_split(
        mols, train_size=0.7, test_size=0.2, valid_size=0.1, return_indices=True
    )

    assert all(isinstance(idx, int) for idx in train)
    assert all(isinstance(idx, int) for idx in valid)
    assert all(isinstance(idx, int) for idx in test)
    assert_equal(len(set(train) | set(valid) | set(test)), len(varied_mols))


def test_maxmin_train_test_split_with_additional_data(varied_mols):
    mols = [Chem.MolFromSmiles(smi) for smi in varied_mols]
    labels = np.ones(len(varied_mols))
    train_mols, test_mols, train_labels, test_labels = maxmin_train_test_split(
        mols, labels
    )
    assert_equal(len(train_mols), len(train_labels))
    assert_equal(len(test_mols), len(test_labels))


def test_maxmin_train_valid_test_split_with_additional_data(varied_mols):
    mols = [Chem.MolFromSmiles(smi) for smi in varied_mols]
    labels = np.ones(len(varied_mols))
    train, valid, test, y_train, y_valid, y_test = maxmin_train_valid_test_split(
        mols, labels
    )
    assert_equal(len(train), len(y_train))
    assert_equal(len(valid), len(y_valid))
    assert_equal(len(test), len(y_test))


def test_maxmin_stratified_train_test_split(mols_list):
    labels = [0] * (len(mols_list) // 2) + [1] * (len(mols_list) // 2)
    mols_train, mols_test, y_train, y_test = maxmin_stratified_train_test_split(
        mols_list, labels, train_size=0.7, test_size=0.3
    )

    assert_equal(len(mols_train), int(0.7 * len(mols_list)))
    assert_equal(len(mols_test), int(0.3 * len(mols_list)))
    assert_equal(len(mols_train) + len(mols_test), len(mols_list))

    assert_equal(len(mols_train), len(y_train))
    assert_equal(len(mols_test), len(y_test))

    assert_allclose(y_train.mean(), 0.5)
    assert_allclose(y_test.mean(), 0.5)
    assert_equal(len(np.concatenate((y_train, y_test))), len(mols_list))


def test_maxmin_stratified_train_valid_test_split(mols_list):
    labels = [0] * (len(mols_list) // 2) + [1] * (len(mols_list) // 2)
    mols_train, mols_valid, mols_test, y_train, y_valid, y_test = (
        maxmin_stratified_train_valid_test_split(
            mols_list, labels, train_size=0.7, valid_size=0.1, test_size=0.2
        )
    )

    assert_equal(len(mols_train), int(0.7 * len(mols_list)))
    assert_equal(len(mols_valid), int(0.1 * len(mols_list)))
    assert_equal(len(mols_test), int(0.2 * len(mols_list)))
    assert_equal(len(mols_train) + len(mols_valid) + len(mols_test), len(mols_list))

    assert_equal(len(mols_train), len(y_train))
    assert_equal(len(mols_valid), len(y_valid))
    assert_equal(len(mols_test), len(y_test))

    assert_allclose(y_train.mean(), 0.5)
    assert_allclose(y_valid.mean(), 0.5)
    assert_allclose(y_test.mean(), 0.5)
    assert_equal(len(np.concatenate((y_train, y_valid, y_test))), len(mols_list))


def test_maxmin_stratified_train_test_split_return_indices(mols_list):
    labels = [0] * (len(mols_list) // 2) + [1] * (len(mols_list) // 2)
    train, test, y_train, y_test = maxmin_stratified_train_test_split(
        mols_list, labels, train_size=0.7, test_size=0.3, return_indices=True
    )

    assert all(isinstance(idx, int) for idx in train)
    assert all(isinstance(idx, int) for idx in test)
    assert_equal(len(set(train) | set(test)), len(mols_list))


def test_maxmin_stratified_train_valid_test_split_returns_indices(mols_list):
    labels = [0] * (len(mols_list) // 2) + [1] * (len(mols_list) // 2)
    train, valid, test, y_train, y_valid, y_test = (
        maxmin_stratified_train_valid_test_split(
            mols_list,
            labels,
            train_size=0.7,
            test_size=0.2,
            valid_size=0.1,
            return_indices=True,
        )
    )

    assert all(isinstance(idx, int) for idx in train)
    assert all(isinstance(idx, int) for idx in valid)
    assert all(isinstance(idx, int) for idx in test)
    assert_equal(len(set(train) | set(valid) | set(test)), len(mols_list))


def test_maxmin_stratified_train_test_split_with_additional_data(mols_list):
    labels = [0] * (len(mols_list) // 2) + [1] * (len(mols_list) // 2)
    extra = ["a"] * len(mols_list)

    train_mols, test_mols, y_train, y_test, extra_train, extra_test = (
        maxmin_stratified_train_test_split(mols_list, labels, extra, test_size=0.2)
    )

    assert_equal(len(train_mols), len(y_train))
    assert_equal(len(test_mols), len(y_test))
    assert_equal(len(train_mols), len(extra_train))
    assert_equal(len(test_mols), len(extra_test))


def test_maxmin_stratified_train_valid_test_split_with_additional_data(mols_list):
    labels = [0] * (len(mols_list) // 2) + [1] * (len(mols_list) // 2)
    extra = ["a"] * len(mols_list)

    (
        train_mols,
        valid_mols,
        test_mols,
        y_train,
        y_valid,
        y_test,
        extra_train,
        extra_valid,
        extra_test,
    ) = maxmin_stratified_train_valid_test_split(
        mols_list, labels, extra, train_size=0.7, valid_size=0.1, test_size=0.2
    )

    assert_equal(len(train_mols), len(y_train))
    assert_equal(len(valid_mols), len(y_valid))
    assert_equal(len(test_mols), len(y_test))

    assert_equal(len(train_mols), len(extra_train))
    assert_equal(len(valid_mols), len(extra_valid))
    assert_equal(len(test_mols), len(extra_test))
