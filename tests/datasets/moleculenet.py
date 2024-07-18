import numpy as np
import pandas as pd

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


def test_load_esol():
    smiles_list, y = load_esol()
    df = load_esol(as_frame=True)

    assert isinstance(smiles_list, list)
    assert all(isinstance(smiles, str) for smiles in smiles_list)
    assert isinstance(y, np.ndarray)
    assert isinstance(df, pd.DataFrame)

    assert len(smiles_list) == 1128
    assert y.shape == (1128,)
    assert np.issubdtype(y.dtype, float)
    assert df.shape == (1128, 2)

    df_smiles = df["SMILES"].tolist()
    df_y = df.drop(columns="SMILES").values.ravel()

    assert smiles_list == df_smiles
    assert np.array_equal(y, df_y)


def test_load_freesolv():
    smiles_list, y = load_freesolv()
    df = load_freesolv(as_frame=True)

    assert isinstance(smiles_list, list)
    assert all(isinstance(smiles, str) for smiles in smiles_list)
    assert isinstance(y, np.ndarray)
    assert isinstance(df, pd.DataFrame)

    assert len(smiles_list) == 642
    assert y.shape == (642,)
    assert np.issubdtype(y.dtype, float)
    assert df.shape == (642, 2)

    df_smiles = df["SMILES"].tolist()
    df_y = df.drop(columns="SMILES").values.ravel()

    assert smiles_list == df_smiles
    assert np.array_equal(y, df_y)


def test_load_lipophilicity():
    smiles_list, y = load_lipophilicity()
    df = load_lipophilicity(as_frame=True)

    assert isinstance(smiles_list, list)
    assert all(isinstance(smiles, str) for smiles in smiles_list)
    assert isinstance(y, np.ndarray)
    assert isinstance(df, pd.DataFrame)

    assert len(smiles_list) == 4200
    assert y.shape == (4200,)
    assert np.issubdtype(y.dtype, float)
    assert df.shape == (4200, 2)

    df_smiles = df["SMILES"].tolist()
    df_y = df.drop(columns="SMILES").values.ravel()

    assert smiles_list == df_smiles
    assert np.array_equal(y, df_y)


def test_load_bace():
    smiles_list, y = load_bace()
    df = load_bace(as_frame=True)

    assert isinstance(smiles_list, list)
    assert all(isinstance(smiles, str) for smiles in smiles_list)
    assert isinstance(y, np.ndarray)
    assert isinstance(df, pd.DataFrame)

    assert len(smiles_list) == 1513
    assert y.shape == (1513,)
    assert np.issubdtype(y.dtype, np.integer)
    assert np.all(np.isin(y, [0, 1]))
    assert df.shape == (1513, 2)

    df_smiles = df["SMILES"].tolist()
    df_y = df.drop(columns="SMILES").values.ravel()

    assert smiles_list == df_smiles
    assert np.array_equal(y, df_y)


def test_load_bbbp():
    smiles_list, y = load_bbbp()
    df = load_bbbp(as_frame=True)

    assert isinstance(smiles_list, list)
    assert all(isinstance(smiles, str) for smiles in smiles_list)
    assert isinstance(y, np.ndarray)
    assert isinstance(df, pd.DataFrame)

    assert len(smiles_list) == 2039
    assert y.shape == (2039,)
    assert np.issubdtype(y.dtype, np.integer)
    assert np.all(np.isin(y, [0, 1]))
    assert df.shape == (2039, 2)

    df_smiles = df["SMILES"].tolist()
    df_y = df.drop(columns="SMILES").values.ravel()

    assert smiles_list == df_smiles
    assert np.array_equal(y, df_y)


def test_load_hiv():
    smiles_list, y = load_hiv()
    df = load_hiv(as_frame=True)

    assert isinstance(smiles_list, list)
    assert all(isinstance(smiles, str) for smiles in smiles_list)
    assert isinstance(y, np.ndarray)
    assert isinstance(df, pd.DataFrame)

    assert len(smiles_list) == 41127
    assert y.shape == (41127,)
    assert np.issubdtype(y.dtype, np.integer)
    assert np.all(np.isin(y, [0, 1]))
    assert df.shape == (41127, 2)

    df_smiles = df["SMILES"].tolist()
    df_y = df.drop(columns="SMILES").values.ravel()

    assert smiles_list == df_smiles
    assert np.array_equal(y, df_y)


def test_load_clintox():
    smiles_list, y = load_clintox()
    df = load_clintox(as_frame=True)

    assert isinstance(smiles_list, list)
    assert all(isinstance(smiles, str) for smiles in smiles_list)
    assert isinstance(y, np.ndarray)
    assert isinstance(df, pd.DataFrame)

    assert len(smiles_list) == 1477
    assert y.shape == (1477, 2)
    assert np.issubdtype(y.dtype, np.integer)
    assert np.all(np.isin(y, [0, 1]))
    assert np.sum(np.isnan(y)) == 0
    assert df.shape == (1477, 3)

    df_smiles = df["SMILES"].tolist()
    df_y = df.drop(columns="SMILES").values

    assert smiles_list == df_smiles
    assert np.array_equal(y, df_y)


def test_load_muv():
    smiles_list, y = load_muv()
    df = load_muv(as_frame=True)

    assert isinstance(smiles_list, list)
    assert all(isinstance(smiles, str) for smiles in smiles_list)
    assert isinstance(y, np.ndarray)
    assert isinstance(df, pd.DataFrame)

    assert len(smiles_list) == 93087
    assert y.shape == (93087, 17)
    assert np.issubdtype(y.dtype, float)
    assert np.all(np.isin(y, [0, 1]) | np.isnan(y))
    assert df.shape == (93087, 18)

    df_smiles = df["SMILES"].tolist()
    df_y = df.drop(columns="SMILES").values

    assert smiles_list == df_smiles
    assert np.array_equal(y, df_y, equal_nan=True)


def test_load_sider():
    smiles_list, y = load_sider()
    df = load_sider(as_frame=True)

    assert isinstance(smiles_list, list)
    assert all(isinstance(smiles, str) for smiles in smiles_list)
    assert isinstance(y, np.ndarray)
    assert isinstance(df, pd.DataFrame)

    assert len(smiles_list) == 1427
    assert y.shape == (1427, 27)
    assert np.issubdtype(y.dtype, np.integer)
    assert np.all(np.isin(y, [0, 1]))
    assert np.sum(np.isnan(y)) == 0
    assert df.shape == (1427, 28)

    df_smiles = df["SMILES"].tolist()
    df_y = df.drop(columns="SMILES").values

    assert smiles_list == df_smiles
    assert np.array_equal(y, df_y)


def test_load_tox21():
    smiles_list, y = load_tox21()
    df = load_tox21(as_frame=True)

    assert isinstance(smiles_list, list)
    assert all(isinstance(smiles, str) for smiles in smiles_list)
    assert isinstance(y, np.ndarray)
    assert isinstance(df, pd.DataFrame)

    assert len(smiles_list) == 7831
    assert y.shape == (7831, 12)
    assert np.issubdtype(y.dtype, float)
    assert np.all(np.isin(y, [0, 1]) | np.isnan(y))
    assert df.shape == (7831, 13)

    df_smiles = df["SMILES"].tolist()
    df_y = df.drop(columns="SMILES").values

    assert smiles_list == df_smiles
    assert np.array_equal(y, df_y, equal_nan=True)


def test_load_toxcast():
    smiles_list, y = load_toxcast()
    df = load_toxcast(as_frame=True)

    assert isinstance(smiles_list, list)
    assert all(isinstance(smiles, str) for smiles in smiles_list)
    assert isinstance(y, np.ndarray)
    assert isinstance(df, pd.DataFrame)

    assert len(smiles_list) == 8576
    assert y.shape == (8576, 617)
    assert np.issubdtype(y.dtype, float)
    assert np.all(np.isin(y, [0, 1]) | np.isnan(y))
    assert df.shape == (8576, 618)

    df_smiles = df["SMILES"].tolist()
    df_y = df.drop(columns="SMILES").values

    assert smiles_list == df_smiles
    assert np.array_equal(y, df_y, equal_nan=True)


def test_load_pcba():
    smiles_list, y = load_pcba()
    df = load_pcba(as_frame=True)

    assert isinstance(smiles_list, list)
    assert all(isinstance(smiles, str) for smiles in smiles_list)
    assert isinstance(y, np.ndarray)
    assert isinstance(df, pd.DataFrame)

    assert len(smiles_list) == 437929
    assert y.shape == (437929, 128)
    assert np.issubdtype(y.dtype, float)
    assert np.all(np.isin(y, [0, 1]) | np.isnan(y))
    assert df.shape == (437929, 129)

    df_smiles = df["SMILES"].tolist()
    df_y = df.drop(columns="SMILES").values

    assert smiles_list == df_smiles
    assert np.array_equal(y, df_y, equal_nan=True)
