import numpy as np
import pytest

from skfp.preprocessing import TiceInsecticidesFilter


@pytest.fixture
def smiles_passing_tice_insecticides() -> list[str]:
    return [
        "O=C(CC1COc2ccccc2O1)NCCc1ccccc1",
        "Cc1cc(C)c(C)c(S(=O)(=O)Nc2ccc(OCC(=O)O)cc2)c1C",
        "O=C(Nc1cccc(Cl)c1)N1CCCC1",
    ]


@pytest.fixture
def smiles_failing_tice_insecticides() -> list[str]:
    return [
        "CNc1nc(N)c([N+](=O)[O-])c(NCCO)n1",
        "CCC(C)n1c(O)c(C(c2ccccn2)c2c(O)n(C(C)CC)c(=S)[nH]c2=O)c(=O)[nH]c1=S",
        "Cn1c(=O)c2c(nc(CN3CCOCC3)n2CCN2CCOCC2)n(C)c1=O",
    ]


@pytest.fixture
def smiles_passing_one_violation_tice_insecticides() -> list[str]:
    return [
        "CCOC(=O)C1CCCN(c2nc3ccccc3nc2C(C#N)C(=O)OC(C)COC)C1",
        "COc1ccc(C2C/C(=C3/C(=O)N=c4ccc(Cl)cc4=C3c3ccccc3)NN2S(C)(=O)=O)cc1",
        "COc1cc(C2C3=C(CC(c4ccco4)CC3=O)Nc3ccccc3N2C(C)=O)cc(OC)c1OC",
    ]


def test_mols_passing_tice_insecticides(
    smiles_passing_tice_insecticides,
):
    mol_filter = TiceInsecticidesFilter()
    smiles_filtered = mol_filter.transform(smiles_passing_tice_insecticides)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert len(smiles_filtered) == len(smiles_passing_tice_insecticides)


def test_mols_failing_tice_insecticides(smiles_failing_tice_insecticides):
    mol_filter = TiceInsecticidesFilter()
    smiles_filtered = mol_filter.transform(smiles_failing_tice_insecticides)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert len(smiles_filtered) == 0


def test_mols_passing_with_violation_tice_insecticides(
    smiles_passing_one_violation_tice_insecticides,
):
    mol_filter = TiceInsecticidesFilter(allow_one_violation=True)
    smiles_filtered = mol_filter.transform(
        smiles_passing_one_violation_tice_insecticides
    )
    assert len(smiles_filtered) == 3

    mol_filter = TiceInsecticidesFilter(allow_one_violation=False)
    smiles_filtered = mol_filter.transform(
        smiles_passing_one_violation_tice_insecticides
    )
    assert len(smiles_filtered) == 0


def test_tice_insecticides_return_indicators(
    smiles_passing_tice_insecticides,
    smiles_failing_tice_insecticides,
    smiles_passing_one_violation_tice_insecticides,
):
    all_smiles = (
        smiles_passing_tice_insecticides
        + smiles_failing_tice_insecticides
        + smiles_passing_one_violation_tice_insecticides
    )

    mol_filter = TiceInsecticidesFilter(return_indicators=True)
    filter_indicators = mol_filter.transform(all_smiles)
    expected_indicators = np.array(
        [True] * len(smiles_passing_tice_insecticides)
        + [False] * len(smiles_failing_tice_insecticides)
        + [False] * len(smiles_passing_one_violation_tice_insecticides),
        dtype=bool,
    )
    assert np.array_equal(filter_indicators, expected_indicators)

    mol_filter = TiceInsecticidesFilter(
        allow_one_violation=True, return_indicators=True
    )
    filter_indicators = mol_filter.transform(all_smiles)
    expected_indicators = np.array(
        [True] * len(smiles_passing_tice_insecticides)
        + [False] * len(smiles_failing_tice_insecticides)
        + [True] * len(smiles_passing_one_violation_tice_insecticides),
        dtype=bool,
    )
    assert np.array_equal(filter_indicators, expected_indicators)


def test_tice_insecticides_parallel(smiles_list):
    mol_filter = TiceInsecticidesFilter()
    mols_filtered_sequential = mol_filter.transform(smiles_list)

    mol_filter = TiceInsecticidesFilter(n_jobs=-1, batch_size=1)
    mols_filtered_parallel = mol_filter.transform(smiles_list)

    assert mols_filtered_sequential == mols_filtered_parallel
