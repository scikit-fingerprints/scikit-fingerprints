from unittest.mock import patch

import numpy as np
import pytest
from numpy.testing import assert_equal
from rdkit import Chem
from rdkit.Chem import Mol

from skfp.model_selection.splitters.pubchem_split import (
    _get_cid_for_smiles,
    _get_earliest_publication_date,
    pubchem_train_test_split,
    pubchem_train_valid_test_split,
)


@pytest.fixture
def get_none() -> None:
    return None


@pytest.fixture
def varied_mols() -> list[str]:
    return [
        "c1ccccc1",
        "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "[Na+].[Cl-]",
        "c1ncccc1[C@@H]2CCCN2C",
        "OC2(C(c1ccc(OC)cc1)CN(C)C)CCCCC2",
        "ClC1=CC2=C(N=C(NC)C[N+]([O-])=C2C3=CC=CC=C3)C=C1",
        "O=C1CC2(CCCC2)CC(=O)N1CCCCN1CCN(c2ncccn2)CC1",
        "CC1OC(C)OC(C)O1",
        r"N\1=C(\c3c(Sc2c/1cccc2)cccc3)N4CCN(CCOCCO)CC4",
    ]


@pytest.fixture
def varied_mols_years() -> list[int | None]:
    return [1860, 1985, 1851, 1851, None, 1983, 1863, 1979, 1882, 1983]


@patch("skfp.model_selection.splitters.pubchem_split.requests.get")
def test_get_cid_for_smiles_with_proper_smiles(mock):
    proper_smiles = "Cc1cc(cc(c1)Oc2ccc(cc2Br)Cc3c4ccnnc4[nH]n3)C#N"
    mock.return_value.status_code = 200
    mock.return_value.json.return_value = {"IdentifierList": {"CID": [25058138]}}

    cid = _get_cid_for_smiles(proper_smiles, 1, verbosity=0)
    assert_equal(cid, "25058138")


@patch("skfp.model_selection.splitters.pubchem_split.requests.get")
def test_get_cid_for_smiles_with_wrong_smiles(mock):
    smiles_that_cannot_be_standarized = "c1ccccc12cCc4"
    mock.return_value.status_code = 400
    mock.return_value.json.return_value = {
        "Fault": {
            "Code": "PUGREST.BadRequest",
            "Message": "Unable to standardize the given structure...",
            "Details": [
                "Record 1: Error: Unable to convert input into a compound object"
            ],
        }
    }

    cid = _get_cid_for_smiles(smiles_that_cannot_be_standarized, 1, verbosity=0)
    assert cid is None


@patch("skfp.model_selection.splitters.pubchem_split.requests.get")
def test_get_cid_for_smiles_with_no_existing_smiles(mock):
    smiles_not_exists = "C[Fe](C)(C)OC(C)=O"
    mock.return_value.status_code = 200
    mock.return_value.json.return_value = {"IdentifierList": {"CID": [0]}}

    cid = _get_cid_for_smiles(smiles_not_exists, 1, verbosity=0)
    assert cid is None


@patch("skfp.model_selection.splitters.pubchem_split.requests.get")
def test_get_earliest_publication_date_proper_cid(mock):
    proper_cid = "24261"
    mock_response = [{"pclid": "204154486", "articlepubdate": "1833-11"}]
    mock.return_value.status_code = 200
    mock.return_value.json.return_value = mock_response

    year = _get_earliest_publication_date(proper_cid, 1)
    assert_equal(year, 1833)


@patch("skfp.model_selection.splitters.pubchem_split.requests.get")
def test_get_earliest_publication_date_none_cid(mock, get_none):
    mock.return_value.status_code = 200
    mock.return_value.json.return_value = {}

    year = _get_earliest_publication_date(get_none, 1)
    assert year is None


@patch("skfp.model_selection.splitters.pubchem_split._get_pubchem_years")
def test_pubchem_train_test_split_default(mock, varied_mols_years, varied_mols):
    mock.return_value = varied_mols_years
    train, test = pubchem_train_test_split(varied_mols, return_indices=False)

    assert_equal(len(train) + len(test), len(varied_mols))
    assert_equal(len(train), 7)
    assert_equal(len(test), 3)


@patch("skfp.model_selection.splitters.pubchem_split._get_pubchem_years")
def test_pubchem_train_valid_test_split_default(mock, varied_mols_years, varied_mols):
    mock.return_value = varied_mols_years
    train, valid, test = pubchem_train_valid_test_split(
        varied_mols, return_indices=False
    )

    assert_equal(len(train) + len(valid) + len(test), len(varied_mols))
    assert_equal(len(train), 8)
    assert_equal(len(valid), 1)
    assert_equal(len(test), 1)


@patch("skfp.model_selection.splitters.pubchem_split._get_pubchem_years")
def test_pubchem_train_test_split_returns_molecules(
    mock, varied_mols_years, varied_mols
):
    mock.return_value = varied_mols_years
    mols = [Chem.MolFromSmiles(s) for s in varied_mols]
    train, test = pubchem_train_test_split(mols, return_indices=False)

    assert all(isinstance(m, Mol) for m in train)
    assert all(isinstance(m, Mol) for m in test)


@patch("skfp.model_selection.splitters.pubchem_split._get_pubchem_years")
def test_pubchem_train_test_split_return_indices(mock, varied_mols_years, varied_mols):
    mock.return_value = varied_mols_years
    mols = [Chem.MolFromSmiles(s) for s in varied_mols]
    train_idx, test_idx = pubchem_train_test_split(mols, return_indices=True)

    assert all(isinstance(idx, int) for idx in train_idx)
    assert all(isinstance(idx, int) for idx in test_idx)


@patch("skfp.model_selection.splitters.pubchem_split._get_pubchem_years")
def test_pubchem_train_valid_test_split_returns_molecules(
    mock, varied_mols_years, varied_mols
):
    mock.return_value = varied_mols_years
    mols = [Chem.MolFromSmiles(s) for s in varied_mols]
    train, valid, test = pubchem_train_valid_test_split(mols, return_indices=False)

    assert all(isinstance(m, Mol) for m in train)
    assert all(isinstance(m, Mol) for m in valid)
    assert all(isinstance(m, Mol) for m in test)


@patch("skfp.model_selection.splitters.pubchem_split._get_pubchem_years")
def test_pubchem_train_valid_test_split_return_indices(
    mock, varied_mols_years, varied_mols
):
    mock.return_value = varied_mols_years
    mols = [Chem.MolFromSmiles(s) for s in varied_mols]
    train_idx, valid_idx, test_idx = pubchem_train_valid_test_split(
        mols, return_indices=True
    )

    assert all(isinstance(idx, int) for idx in train_idx)
    assert all(isinstance(idx, int) for idx in valid_idx)
    assert all(isinstance(idx, int) for idx in test_idx)


@patch("skfp.model_selection.splitters.pubchem_split._get_pubchem_years")
def test_pubchem_train_valid_test_split_not_found_behaviour_train(
    mock, varied_mols_years, varied_mols
):
    mock.return_value = varied_mols_years
    train, valid, test = pubchem_train_valid_test_split(
        varied_mols,
        not_found_behavior="train",
        train_size=0.6,
        valid_size=0.2,
        test_size=0.2,
    )

    assert_equal(len(train) + len(valid) + len(test), len(varied_mols_years))
    assert_equal(len(train), 7)
    assert_equal(len(valid), 2)
    assert_equal(len(test), 1)


@patch("skfp.model_selection.splitters.pubchem_split._get_pubchem_years")
def test_pubchem_train_test_split_not_found_behaviour_train(
    mock, varied_mols_years, varied_mols
):
    mock.return_value = varied_mols_years
    train, test = pubchem_train_test_split(varied_mols, not_found_behavior="train")

    assert_equal(len(train) + len(test), len(varied_mols))
    assert_equal(len(train), 8)
    assert_equal(len(test), 2)


@patch("skfp.model_selection.splitters.pubchem_split._get_pubchem_years")
def test_pubchem_train_valid_test_split_not_found_behaviour_remove(
    mock, varied_mols_years, varied_mols
):
    mock.return_value = varied_mols_years
    mols_with_years = sum(bool(y) for y in varied_mols_years)
    train, valid, test = pubchem_train_valid_test_split(
        varied_mols,
        not_found_behavior="remove",
        train_size=0.6,
        valid_size=0.2,
        test_size=0.2,
    )

    assert_equal(len(train) + len(valid) + len(test), mols_with_years)
    assert_equal(len(train), 6)
    assert_equal(len(valid), 2)
    assert_equal(len(test), 1)


@patch("skfp.model_selection.splitters.pubchem_split._get_pubchem_years")
def test_pubchem_train_test_split_with_additional_data(
    mock, varied_mols_years, varied_mols
):
    mock.return_value = varied_mols_years
    labels = np.ones(len(varied_mols))
    train_mols, test_mols, train_labels, test_labels = pubchem_train_test_split(
        varied_mols, labels
    )

    assert_equal(len(train_mols), len(train_labels))
    assert_equal(len(test_mols), len(test_labels))


@patch("skfp.model_selection.splitters.pubchem_split._get_pubchem_years")
def test_pubchem_train_valid_test_split_with_additional_data(
    mock, varied_mols_years, varied_mols
):
    mock.return_value = varied_mols_years
    labels = np.ones(len(varied_mols))
    train, valid, test, y_train, y_valid, y_test = pubchem_train_valid_test_split(
        varied_mols, labels
    )

    assert_equal(len(train), len(y_train))
    assert_equal(len(valid), len(y_valid))
    assert_equal(len(test), len(y_test))
