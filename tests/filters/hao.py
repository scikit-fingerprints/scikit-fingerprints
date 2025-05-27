import numpy as np
import pytest

from skfp.filters import HaoFilter


@pytest.fixture
def smiles_passing_hao() -> list[str]:
    return [
        "CCOC(=O)Nc1ccc(C(=O)C=Cc2ccc(N(CC)CC)cc2)cc1",
        "CN(C)c1ccc(C=Cc2cc[n+](C)c3ccccc23)cc1",
        "c1cnc2c(c1)ccc1cccnc12",
    ]


@pytest.fixture
def smiles_failing_hao() -> list[str]:
    return [
        "O=C(NN=Cc1ccc(Cl)cc1Cl)c1ccccc1SSc1ccccc1C(=O)NN=Cc1ccc(Cl)cc1Cl",
        "O=S1(=O)OC(c2cc(Br)c(O)c(Br)c2)(c2cc(Br)c(O)c(Br)c2)c2c(Br)c(Br)c(Br)c(Br)c21",
        "Cc1cc2c(C(C)C)c(O)c(O)c(C=NCc3ccoc3)c2c(O)c1-c1c(C)cc2c(C(C)C)c(O)c(O)c(C=NCc3ccco3)c2c1O",
    ]


@pytest.fixture
def smiles_passing_one_violation_hao() -> list[str]:
    return [
        "Nc1cc([As]=[As]c2ccc(O)c(N)c2)ccc1O",
        "Cn1c(SSc2ccc(-c3cccnc3)n2C)ccc1-c1cccnc1",
        "COc1c2c(cc3c1OCO3)C13C=CC(OC)CC1N(CC3)C2.[O-][Cl+3]([O-])([O-])O",
    ]


def test_mols_passing_hao(
    smiles_passing_hao,
):
    mol_filter = HaoFilter()
    smiles_filtered = mol_filter.transform(smiles_passing_hao)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert len(smiles_filtered) == len(smiles_passing_hao)


def test_mols_failing_hao(smiles_failing_hao):
    mol_filter = HaoFilter()
    smiles_filtered = mol_filter.transform(smiles_failing_hao)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert len(smiles_filtered) == 0


def test_mols_passing_with_violation_hao(
    smiles_passing_one_violation_hao,
):
    mol_filter = HaoFilter(allow_one_violation=True)
    smiles_filtered = mol_filter.transform(smiles_passing_one_violation_hao)
    assert len(smiles_filtered) == 3

    mol_filter = HaoFilter(allow_one_violation=False)
    smiles_filtered = mol_filter.transform(smiles_passing_one_violation_hao)
    assert len(smiles_filtered) == 0


def test_hao_return_indicators(
    smiles_passing_hao,
    smiles_failing_hao,
    smiles_passing_one_violation_hao,
):
    all_smiles = (
        smiles_passing_hao + smiles_failing_hao + smiles_passing_one_violation_hao
    )

    mol_filter = HaoFilter(return_indicators=True)
    filter_indicators = mol_filter.transform(all_smiles)
    expected_indicators = np.array(
        [True] * len(smiles_passing_hao)
        + [False] * len(smiles_failing_hao)
        + [False] * len(smiles_passing_one_violation_hao),
        dtype=bool,
    )
    assert np.array_equal(filter_indicators, expected_indicators)

    mol_filter = HaoFilter(allow_one_violation=True, return_indicators=True)
    filter_indicators = mol_filter.transform(all_smiles)
    expected_indicators = np.array(
        [True] * len(smiles_passing_hao)
        + [False] * len(smiles_failing_hao)
        + [True] * len(smiles_passing_one_violation_hao),
        dtype=bool,
    )
    assert np.array_equal(filter_indicators, expected_indicators)


def test_hao_parallel(smiles_list):
    mol_filter = HaoFilter()
    mols_filtered_sequential = mol_filter.transform(smiles_list)

    mol_filter = HaoFilter(n_jobs=-1, batch_size=1)
    mols_filtered_parallel = mol_filter.transform(smiles_list)

    assert mols_filtered_sequential == mols_filtered_parallel


def test_hao_transform_x_y(smiles_passing_hao, smiles_failing_hao):
    all_smiles = smiles_passing_hao + smiles_failing_hao
    labels = np.array([1] * len(smiles_passing_hao) + [0] * len(smiles_failing_hao))

    filt = HaoFilter()
    mols, labels_filt = filt.transform_x_y(all_smiles, labels)
    assert len(mols) == len(smiles_passing_hao)
    assert np.all(labels_filt == 1)

    filt = HaoFilter(return_indicators=True)
    indicators, labels_filt = filt.transform_x_y(all_smiles, labels)
    assert np.sum(indicators) == len(smiles_passing_hao)
    assert np.array_equal(indicators, labels_filt)
