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


def get_dataset_names() -> list[str]:
    return [
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


def test_load_moleculenet_benchmark():
    dataset_names = get_dataset_names()
    benchmark_full = load_moleculenet_benchmark(as_frames=True)
    benchmark_names = [name for name, df in benchmark_full]
    assert benchmark_names == dataset_names


def test_load_moleculenet_benchmark_subset():
    dataset_names = ["ESOL", "SIDER", "BACE"]
    benchmark_full = load_moleculenet_benchmark(subset=dataset_names, as_frames=True)
    benchmark_names = [name for name, df in benchmark_full]
    assert benchmark_names == dataset_names


def test_load_moleculenet_benchmark_wrong_subset():
    dataset_names = ["ESOL", "Nonexistent"]
    with pytest.raises(ValueError) as exc_info:
        load_moleculenet_benchmark(subset=dataset_names, as_frames=True)

    assert "Dataset name 'Nonexistent' not recognized" in str(exc_info)


@pytest.mark.parametrize("dataset_name", get_dataset_names())
def test_load_ogb_splits(dataset_name):
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


@pytest.mark.parametrize("dataset_name", get_dataset_names())
def test_load_ogb_splits_as_dict(dataset_name):
    train, valid, test = load_ogb_splits(dataset_name)
    split_idxs = load_ogb_splits(dataset_name, as_dict=True)
    assert isinstance(split_idxs, dict)
    assert set(split_idxs.keys()) == {"train", "valid", "test"}
    assert split_idxs["train"] == train
    assert split_idxs["valid"] == valid
    assert split_idxs["test"] == test


@pytest.mark.parametrize(
    "dataset_name, dataset_length",
    [
        ("ESOL", 1128),
        ("FreeSolv", 642),
        ("Lipophilicity", 4200),
        ("BACE", 1513),
        ("BBBP", 2039),
        ("HIV", 41127),
        ("ClinTox", 1477),
        ("MUV", 93087),
        ("SIDER", 1427),
        ("Tox21", 7831),
        ("ToxCast", 8576),
        ("PCBA", 437929),
    ],
)
def test_load_ogb_splits_lengths(dataset_name, dataset_length):
    train, valid, test = load_ogb_splits(dataset_name)
    loaded_length = len(train) + len(valid) + len(test)
    assert loaded_length == dataset_length


def test_load_ogb_splits_nonexistent_dataset():
    with pytest.raises(InvalidParameterError) as error:
        load_ogb_splits("nonexistent")

    assert str(error.value).startswith(
        "The 'dataset_name' parameter of load_ogb_splits must be a str among"
    )


@pytest.mark.parametrize(
    "dataset_name, load_func, expected_length, num_tasks, task_type",
    [
        ("ESOL", load_esol, 1128, 1, "regression"),
        ("FreeSolv", load_freesolv, 642, 1, "regression"),
        ("Lipophilicity", load_lipophilicity, 4200, 1, "regression"),
        ("BACE", load_bace, 1513, 1, "binary_classification"),
        ("BBBP", load_bbbp, 2039, 1, "binary_classification"),
        ("HIV", load_hiv, 41127, 1, "binary_classification"),
        ("ClinTox", load_clintox, 1477, 2, "binary_classification"),
        ("MUV", load_muv, 93087, 17, "binary_classification"),
        ("SIDER", load_sider, 1427, 27, "binary_classification"),
        ("Tox21", load_tox21, 7831, 12, "binary_classification"),
        ("ToxCast", load_toxcast, 8576, 617, "binary_classification"),
        ("PCBA", load_pcba, 437929, 128, "binary_classification"),
    ],
)
def test_load_dataset(dataset_name, load_func, expected_length, num_tasks, task_type):
    smiles_list, y = load_func()
    df = load_func(as_frame=True)
    run_basic_dataset_checks(
        smiles_list,
        y,
        df,
        expected_length=expected_length,
        num_tasks=num_tasks,
        task_type=task_type,
    )
