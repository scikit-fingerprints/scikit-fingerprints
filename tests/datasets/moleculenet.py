import pytest
from sklearn.utils._param_validation import InvalidParameterError

from skfp.datasets.moleculenet import (
    load_bace,
    load_bbbp,
    load_clintox,
    load_esol,
    load_freesolv,
    load_hiv,
    load_lipophilicity,
    load_moleculenet_benchmark,
    load_muv,
    load_ogb_splits,
    load_pcba,
    load_sider,
    load_tox21,
    load_toxcast,
)
from tests.datasets.test_utils import run_basic_dataset_checks


def test_load_moleculenet_benchmark():
    dataset_names = [
        "ESOL",
        "FreeSolv",
        "Lipophilicity",
        "BACE",
        "BBBP",
        "HIV",
        "ClinTox",
        "MUV",
        "SIDER",
        "Tox21",
        "ToxCast",
        "PCBA",
    ]
    benchmark_full = load_moleculenet_benchmark(as_frames=True)
    benchmark_names = [name for name, df in benchmark_full]
    assert benchmark_names == dataset_names


def test_load_ogb_splits():
    dataset_names = [
        "ESOL",
        "FreeSolv",
        "Lipophilicity",
        "BACE",
        "BBBP",
        "HIV",
        "ClinTox",
        "MUV",
        "SIDER",
        "Tox21",
        "ToxCast",
        "PCBA",
    ]
    for dataset_name in dataset_names:
        train, valid, test = load_ogb_splits(dataset_name)
        assert isinstance(train, list)
        assert len(train) > 0
        assert all(isinstance(idx, int) for idx in train)

        assert isinstance(valid, list)
        assert len(valid) > 0
        assert all(isinstance(idx, int) for idx in valid)

        assert isinstance(test, list)
        assert len(test) > 0
        assert all(isinstance(idx, int) for idx in test)

        assert len(train) > len(valid)
        assert len(train) > len(test)


def test_load_ogb_splits_as_dict():
    dataset_names = [
        "ESOL",
        "FreeSolv",
        "Lipophilicity",
        "BACE",
        "BBBP",
        "HIV",
        "ClinTox",
        "MUV",
        "SIDER",
        "Tox21",
        "ToxCast",
        "PCBA",
    ]
    for dataset_name in dataset_names:
        train, valid, test = load_ogb_splits(dataset_name)
        split_idxs = load_ogb_splits(dataset_name, as_dict=True)
        assert isinstance(split_idxs, dict)
        assert set(split_idxs.keys()) == {"train", "valid", "test"}
        assert split_idxs["train"] == train
        assert split_idxs["valid"] == valid
        assert split_idxs["test"] == test


def test_load_ogb_splits_lengths():
    dataset_lengths = {
        "ESOL": 1128,
        "FreeSolv": 642,
        "Lipophilicity": 4200,
        "BACE": 1513,
        "BBBP": 2039,
        "HIV": 41127,
        "ClinTox": 1477,
        "MUV": 93087,
        "SIDER": 1427,
        "Tox21": 7831,
        "ToxCast": 8576,
        "PCBA": 437929,
    }
    for dataset_name, expected_length in dataset_lengths.items():
        train, valid, test = load_ogb_splits(dataset_name)
        loaded_length = len(train) + len(valid) + len(test)
        assert loaded_length == expected_length


def test_load_ogb_splits_nonexistent_dataset():
    with pytest.raises(InvalidParameterError) as error:
        load_ogb_splits("nonexistent")

    assert str(error.value).startswith(
        "The 'dataset_name' parameter of load_ogb_splits must be a str among"
    )


def test_load_esol():
    smiles_list, y = load_esol()
    df = load_esol(as_frame=True)
    run_basic_dataset_checks(
        smiles_list,
        y,
        df,
        expected_length=1128,
        num_tasks=1,
        task_type="regression",
    )


def test_load_freesolv():
    smiles_list, y = load_freesolv()
    df = load_freesolv(as_frame=True)
    run_basic_dataset_checks(
        smiles_list,
        y,
        df,
        expected_length=642,
        num_tasks=1,
        task_type="regression",
    )


def test_load_lipophilicity():
    smiles_list, y = load_lipophilicity()
    df = load_lipophilicity(as_frame=True)
    run_basic_dataset_checks(
        smiles_list,
        y,
        df,
        expected_length=4200,
        num_tasks=1,
        task_type="regression",
    )


def test_load_bace():
    smiles_list, y = load_bace()
    df = load_bace(as_frame=True)
    run_basic_dataset_checks(
        smiles_list,
        y,
        df,
        expected_length=1513,
        num_tasks=1,
        task_type="binary_classification",
    )


def test_load_bbbp():
    smiles_list, y = load_bbbp()
    df = load_bbbp(as_frame=True)
    run_basic_dataset_checks(
        smiles_list,
        y,
        df,
        expected_length=2039,
        num_tasks=1,
        task_type="binary_classification",
    )


def test_load_hiv():
    smiles_list, y = load_hiv()
    df = load_hiv(as_frame=True)
    run_basic_dataset_checks(
        smiles_list,
        y,
        df,
        expected_length=41127,
        num_tasks=1,
        task_type="binary_classification",
    )


def test_load_clintox():
    smiles_list, y = load_clintox()
    df = load_clintox(as_frame=True)
    run_basic_dataset_checks(
        smiles_list,
        y,
        df,
        expected_length=1477,
        num_tasks=2,
        task_type="binary_classification",
    )


def test_load_muv():
    smiles_list, y = load_muv()
    df = load_muv(as_frame=True)
    run_basic_dataset_checks(
        smiles_list,
        y,
        df,
        expected_length=93087,
        num_tasks=17,
        task_type="binary_classification",
    )


def test_load_sider():
    smiles_list, y = load_sider()
    df = load_sider(as_frame=True)
    run_basic_dataset_checks(
        smiles_list,
        y,
        df,
        expected_length=1427,
        num_tasks=27,
        task_type="binary_classification",
    )


def test_load_tox21():
    smiles_list, y = load_tox21()
    df = load_tox21(as_frame=True)
    run_basic_dataset_checks(
        smiles_list,
        y,
        df,
        expected_length=7831,
        num_tasks=12,
        task_type="binary_classification",
    )


def test_load_toxcast():
    smiles_list, y = load_toxcast()
    df = load_toxcast(as_frame=True)
    run_basic_dataset_checks(
        smiles_list,
        y,
        df,
        expected_length=8576,
        num_tasks=617,
        task_type="binary_classification",
    )


def test_load_pcba():
    smiles_list, y = load_pcba()
    df = load_pcba(as_frame=True)
    run_basic_dataset_checks(
        smiles_list,
        y,
        df,
        expected_length=437929,
        num_tasks=128,
        task_type="binary_classification",
    )
