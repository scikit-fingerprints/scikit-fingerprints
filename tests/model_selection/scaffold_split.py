import pytest
from rdkit import Chem

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
        "C1CC1.OCC",
        "C1CCC1.OCC",
        "CC.OCC",
    ]

    return [Chem.MolFromSmiles(smiles) for smiles in all_smiles]


@pytest.fixture
def disconnected_molecules() -> list[str]:
    disconnected_smiles: list[str] = ["C1CC1.OCC", "C1CCC1.OCC", "CC.OCC"]

    return [Chem.MolFromSmiles(smiles) for smiles in disconnected_smiles]


@pytest.fixture
def no_ring_molecules() -> list[str]:
    no_ring_smiles: list[str] = ["CCO", "CCN", "CCC", "CCCl", "CCBr"]

    return [Chem.MolFromSmiles(smiles) for smiles in no_ring_smiles]


@pytest.fixture
def benzodiazepines() -> list[str]:
    benzodiazepines_smiles: list[str] = [
        "C1CN=C(C2=CC=CC=C2)N=C1",
        "C1CN=C(C2=CC=CC=C2F)N=C1",
        "C1CN=C(C2=CC=CC=C2Cl)N=C1",
    ]

    return [Chem.MolFromSmiles(smiles) for smiles in benzodiazepines_smiles]


@pytest.fixture
def monosaccharides() -> list[str]:
    monosaccharides_smiles: list[str] = [
        "C(C1C(C(C(C(O1)O)O)O)O)O",
        "C(C(C1C(C(C(C(O1)O)O)O)O)O)O",
        "C(C(C(C(C(C=O)O)O)O)O)O",
        "C(C(C(C(C(CO)O)O)O)O)O",
        "C(C(C(C(C(CO)O)O)O)O)O",
    ]

    return [Chem.MolFromSmiles(smiles) for smiles in monosaccharides_smiles]


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

    assert len(train_split) < len(test_split)
    assert len(train_split) == 7
    assert len(test_split) == 3


def test_train_split_larger_than_valid_and_test_splits(all_molecules):
    train_split, valid_split, test_split = scaffold_train_valid_test_split(
        all_molecules, train_size=0.7, valid_size=0.2, test_size=0.1
    )

    assert len(train_split) < len(valid_split)
    assert len(valid_split) < len(test_split)
    assert len(train_split) == 7
    assert len(valid_split) == 2
    assert len(test_split) == 1


def test_scaffold_creation_total_count(all_molecules):
    scaffolds = _create_scaffolds(all_molecules)
    assert len(scaffolds) <= len(all_molecules)


def test_disconnected_graphs(disconnected_molecules):
    scaffolds = _create_scaffolds(disconnected_molecules)
    assert len(scaffolds) == 1


def test_no_ring_molecules(no_ring_molecules):
    scaffolds = _create_scaffolds(no_ring_molecules)
    assert len(scaffolds) == 1


def test_scaffold_count_for_benzodiazepines(benzodiazepines):
    scaffolds = _create_scaffolds(benzodiazepines)
    assert len(scaffolds) == 1


def test_scaffold_count_for_monosaccharides(monosaccharides):
    scaffolds = _create_scaffolds(monosaccharides)
    assert len(scaffolds) == 1
