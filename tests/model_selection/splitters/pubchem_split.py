from typing import Union
from unittest.mock import patch

import pytest
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
def varied_mols_years() -> list[Union[int, None]]:
    return [1860, 1985, 1851, 1851, None, 1983, 1863, 1979, 1882, 1983]


@patch("skfp.model_selection.splitters.pubchem_split.requests.get")
def test_get_cid_for_smiles_with_proper_smiles(mock):
    proper_smiles = "Cc1cc(cc(c1)Oc2ccc(cc2Br)Cc3c4ccnnc4[nH]n3)C#N"
    mock.return_value.status_code = 200
    mock.return_value.json.return_value = {"IdentifierList": {"CID": [25058138]}}

    cid = _get_cid_for_smiles(proper_smiles, 1)

    assert cid == "25058138"


@patch("skfp.model_selection.splitters.pubchem_split.requests.get")
def test_get_cid_for_smiles_with_wrong_smiles(mock):
    smiles_that_cannot_be_standarized = "c1ccccc12cCc4"
    mock.return_value.status_code = 400
    mock.return_value.json.return_value = {
        "Fault": {
            "Code": "PUGREST.BadRequest",
            "Message": "Unable to standardize the given structure - perhaps some special characters need to be escaped \
            or data packed in a MIME form?",
            "Details": [
                "error: ",
                "status: 400",
                "output: Caught ncbi::CException: Standardization failed",
                "Output Log:",
                "Record 1: Warning: Cactvs Ensemble cannot be created from input string",
                "Record 1: Error: Unable to convert input into a compound object",
                "",
                "",
            ],
        }
    }

    cid = _get_cid_for_smiles(smiles_that_cannot_be_standarized, 1)
    assert cid is None


@patch("skfp.model_selection.splitters.pubchem_split.requests.get")
def test_get_cid_for_smiles_with_no_existing_smiles(mock):
    smiles_that_not_exists_in_pubchem = "C[Fe](C)(C)OC(C)=O"
    mock.return_value.status_code = 200
    mock.return_value.json.return_value = {"IdentifierList": {"CID": [0]}}
    cid = _get_cid_for_smiles(smiles_that_not_exists_in_pubchem, 1)
    assert cid is None


@patch("skfp.model_selection.splitters.pubchem_split.requests.get")
def test_get_earliest_publication_date_proper_cid(mock):
    proper_pubchem_cid = "24261"
    mock_response = [{"pclid": "204154486", "articlepubdate": "1833-11"}]
    mock.return_value.status_code = 200
    mock.return_value.json.return_value = mock_response
    year = _get_earliest_publication_date(proper_pubchem_cid, 1)
    assert year == 1833


@patch("skfp.model_selection.splitters.pubchem_split.requests.get")
def test_get_earliest_publication_date_none_cid(mock, get_none):
    mock.return_value.status_code = 200
    mock.return_value.json.return_value = {}
    year = _get_earliest_publication_date(get_none, 1)
    assert year is None


@patch("skfp.model_selection.splitters.pubchem_split._get_pubchem_years")
def test_pubchem_train_test_split_default(mock, varied_mols_years, varied_mols):
    mock.return_value = varied_mols_years
    train_split, test_split = pubchem_train_test_split(
        varied_mols, return_indices=False
    )

    assert len(train_split) + len(test_split) == len(varied_mols)
    assert len(train_split) == 7
    assert len(test_split) == 3


@patch("skfp.model_selection.splitters.pubchem_split._get_pubchem_years")
def test_pubchem_train_valid_test_split_default(mock, varied_mols_years, varied_mols):
    mock.return_value = varied_mols_years
    train_split, valid_split, test_split = pubchem_train_valid_test_split(
        varied_mols, return_indices=False
    )

    assert len(train_split) + len(valid_split) + len(test_split) == len(varied_mols)
    assert len(train_split) == 8
    assert len(valid_split) == 1
    assert len(test_split) == 1


@patch("skfp.model_selection.splitters.pubchem_split._get_pubchem_years")
def test_pubchem_train_test_split_returns_molecules(
    mock, varied_mols_years, varied_mols
):
    mock.return_value = varied_mols_years
    mols = [Chem.MolFromSmiles(smiles) for smiles in varied_mols]
    train_set, test_set = pubchem_train_test_split(mols, return_indices=False)

    assert all(isinstance(train, Mol) for train in train_set)
    assert all(isinstance(test, Mol) for test in test_set)


@patch("skfp.model_selection.splitters.pubchem_split._get_pubchem_years")
def test_pubchem_train_test_split_return_indices(mock, varied_mols_years, varied_mols):
    mock.return_value = varied_mols_years
    mols = [Chem.MolFromSmiles(smiles) for smiles in varied_mols]
    train_idxs, test_idxs = pubchem_train_test_split(mols, return_indices=True)

    assert all(isinstance(idx, int) for idx in train_idxs)
    assert all(isinstance(idx, int) for idx in test_idxs)


@patch("skfp.model_selection.splitters.pubchem_split._get_pubchem_years")
def test_pubchem_train_valid_test_split_returns_molecules(
    mock, varied_mols_years, varied_mols
):
    mock.return_value = varied_mols_years
    mols = [Chem.MolFromSmiles(smiles) for smiles in varied_mols]
    train_set, valid_set, test_set = pubchem_train_valid_test_split(
        mols, return_indices=False
    )

    assert all(isinstance(train, Mol) for train in train_set)
    assert all(isinstance(valid, Mol) for valid in valid_set)
    assert all(isinstance(test, Mol) for test in test_set)


@patch("skfp.model_selection.splitters.pubchem_split._get_pubchem_years")
def test_pubchem_train_valid_test_split_return_indices(
    mock, varied_mols_years, varied_mols
):
    mock.return_value = varied_mols_years
    mols = [Chem.MolFromSmiles(smiles) for smiles in varied_mols]
    train_idxs, valid_idxs, test_idxs = pubchem_train_valid_test_split(
        mols, return_indices=True
    )

    assert all(isinstance(idx, int) for idx in train_idxs)
    assert all(isinstance(idx, int) for idx in valid_idxs)
    assert all(isinstance(idx, int) for idx in test_idxs)


@patch("skfp.model_selection.splitters.pubchem_split._get_pubchem_years")
def test_pubchem_train_valid_test_split_not_found_behaviour_train(
    mock, varied_mols_years, varied_mols
):
    mock.return_value = varied_mols_years
    train_set, valid_set, test_set = pubchem_train_valid_test_split(
        varied_mols,
        return_indices=False,
        not_found_behavior="train",
        train_size=0.6,
        valid_size=0.2,
        test_size=0.2,
    )
    assert len(train_set) + len(valid_set) + len(test_set) == len(varied_mols_years)
    assert len(train_set) == 7
    assert len(valid_set) == 2
    assert len(test_set) == 1


@patch("skfp.model_selection.splitters.pubchem_split._get_pubchem_years")
def test_pubchem_train_test_split_not_found_behaviour_train(
    mock, varied_mols_years, varied_mols
):
    mock.return_value = varied_mols_years
    train_split, test_split = pubchem_train_test_split(
        varied_mols, return_indices=False, not_found_behavior="train"
    )

    assert len(train_split) + len(test_split) == len(varied_mols)
    assert len(train_split) == 8
    assert len(test_split) == 2


@patch("skfp.model_selection.splitters.pubchem_split._get_pubchem_years")
def test_pubchem_train_valid_test_split_not_found_behaviour_remove(
    mock, varied_mols_years, varied_mols
):
    mock.return_value = varied_mols_years
    mols_with_years = sum(bool(y) for y in varied_mols_years)
    train_set, valid_set, test_set = pubchem_train_valid_test_split(
        varied_mols,
        return_indices=False,
        not_found_behavior="remove",
        train_size=0.6,
        valid_size=0.2,
        test_size=0.2,
    )
    assert len(train_set) + len(valid_set) + len(test_set) == mols_with_years
    assert len(train_set) == 6
    assert len(valid_set) == 2
    assert len(test_set) == 1
