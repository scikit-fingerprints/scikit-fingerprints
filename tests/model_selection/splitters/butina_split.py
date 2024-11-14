import pytest
from rdkit import Chem
from rdkit.Chem import Mol

from skfp.model_selection.splitters.butina_split import (
    _create_clusters,
    butina_train_test_split,
    butina_train_valid_test_split,
)
from skfp.preprocessing import MolFromSmilesTransformer


@pytest.fixture
def varied_mols() -> list[str]:
    # those molecules are quite varied, so they get high Tanimoto distance
    # and thus constitute separate centroids for Taylor-Butina clustering
    return [
        # benzene
        "c1ccccc1",
        # ibuprofen
        "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
        # caffeine
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        # sodium chloride (salt)
        "[Na+].[Cl-]",
        # nicotine
        "c1ncccc1[C@@H]2CCCN2C",
        # Venlafaxine
        "OC2(C(c1ccc(OC)cc1)CN(C)C)CCCCC2",
        # Chlordiazepoxide
        "ClC1=CC2=C(N=C(NC)C[N+]([O-])=C2C3=CC=CC=C3)C=C1",
        # Buspirone
        "O=C1CC2(CCCC2)CC(=O)N1CCCCN1CCN(c2ncccn2)CC1",
        # Paraldehyde
        "CC1OC(C)OC(C)O1",
        # Quetiapine
        r"N\1=C(\c3c(Sc2c/1cccc2)cccc3)N4CCN(CCOCCO)CC4",
    ]


def test_butina_train_test_split_default(varied_mols):
    train, test = butina_train_test_split(varied_mols)
    assert len(train) == 8
    assert len(test) == 2


def test_butina_train_test_split_custom_sizes(varied_mols):
    train, test = butina_train_test_split(varied_mols, train_size=0.7, test_size=0.3)
    assert len(train) == 7
    assert len(test) == 3


def test_train_test_split_total_molecule_count(varied_mols):
    train_split, test_split = butina_train_test_split(
        varied_mols, train_size=0.8, test_size=0.2
    )
    assert len(train_split) + len(test_split) == len(varied_mols)
    assert len(train_split) == 8
    assert len(test_split) == 2


def test_train_valid_test_split_total_molecule_count(varied_mols):
    train_split, valid_split, test_split = butina_train_valid_test_split(
        varied_mols, train_size=0.8, valid_size=0.1, test_size=0.1
    )

    assert len(train_split) + len(valid_split) + len(test_split) == len(varied_mols)
    assert len(train_split) == 8
    assert len(valid_split) == 1
    assert len(test_split) == 1


def test_test_split_smaller_than_train_split(varied_mols):
    train_split, test_split = butina_train_test_split(
        varied_mols, train_size=0.7, test_size=0.3
    )

    assert len(train_split) > len(test_split)
    assert len(train_split) == 7
    assert len(test_split) == 3


def test_train_split_larger_than_valid_and_test_splits(varied_mols):
    train_split, valid_split, test_split = butina_train_valid_test_split(
        varied_mols, train_size=0.7, valid_size=0.2, test_size=0.1
    )

    assert len(train_split) > len(valid_split)
    assert len(valid_split) > len(test_split)
    assert len(train_split) == 7
    assert len(valid_split) == 2
    assert len(test_split) == 1


def test_butina_creation_total_count(varied_mols):
    butinas = _create_clusters(varied_mols)
    assert len(butinas) <= len(varied_mols)


def test_butina_count_for_benzodiazepines():
    smiles_list = [
        "C1CN=C(C2=CC=CC=C2)N=C1",
        "C1CN=C(C2=CC=CC=C2F)N=C1",
        "C1CN=C(C2=CC=CC=C2Cl)N=C1",
    ]
    mols = MolFromSmilesTransformer().transform(smiles_list)

    butinas = _create_clusters(mols)
    assert len(butinas) == 1


def test_butina_train_test_split_returns_molecules(varied_mols):
    mols = [Chem.MolFromSmiles(smiles) for smiles in varied_mols]
    train_set, test_set = butina_train_test_split(mols, return_indices=False)

    assert all(isinstance(train, Mol) for train in train_set)
    assert all(isinstance(test, Mol) for test in test_set)


def test_butina_train_test_split_return_indices(varied_mols):
    mols = [Chem.MolFromSmiles(smiles) for smiles in varied_mols]
    train_idxs, test_idxs = butina_train_test_split(mols, return_indices=True)

    assert all(isinstance(idx, int) for idx in train_idxs)
    assert all(isinstance(idx, int) for idx in test_idxs)


def test_butina_train_valid_test_split_returns_molecules(varied_mols):
    mols = [Chem.MolFromSmiles(smiles) for smiles in varied_mols]
    train_set, valid_set, test_set = butina_train_valid_test_split(
        mols, return_indices=False
    )

    assert all(isinstance(train, Mol) for train in train_set)
    assert all(isinstance(valid, Mol) for valid in valid_set)
    assert all(isinstance(test, Mol) for test in test_set)


def test_butina_train_valid_test_split_return_indices(varied_mols):
    mols = [Chem.MolFromSmiles(smiles) for smiles in varied_mols]
    train_idxs, valid_idxs, test_idxs = butina_train_valid_test_split(
        mols, return_indices=True
    )

    assert all(isinstance(idx, int) for idx in train_idxs)
    assert all(isinstance(idx, int) for idx in valid_idxs)
    assert all(isinstance(idx, int) for idx in test_idxs)


def test_empty_train_subset_raises_an_error_train_test():
    smiles_list = ["C1CCCC(C2CC2)CC1"]

    with pytest.raises(
        ValueError,
        match="the resulting train set will be empty",
    ):
        butina_train_test_split(data=smiles_list)


def test_empty_train_subset_raises_an_error_train_valid_test():
    smiles_list = ["C1CCCC(C2CC2)CC1", "c1n[nH]cc1C1CCCCCC1"]

    with pytest.raises(
        ValueError,
        match="one of the sets will be empty",
    ):
        butina_train_valid_test_split(data=smiles_list)


def test_approximate_butina_split(mols_list):
    train_split, test_split = butina_train_test_split(mols_list, approximate=True)
    assert len(train_split) >= len(test_split)

    train_split, valid_split, test_split = butina_train_valid_test_split(
        mols_list, approximate=True
    )
    assert len(train_split) >= len(valid_split)
    assert len(train_split) >= len(test_split)
