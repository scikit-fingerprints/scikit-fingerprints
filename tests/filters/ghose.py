import numpy as np
import pytest

from skfp.filters import GhoseFilter


@pytest.fixture
def smiles_passing_ghose() -> list[str]:
    return [
        "CC(=O)C1=C(O)C(=O)N(CCc2c[nH]c3ccccc23)C1c1ccc(C)cc1",
        r"CC(=O)C1C(=O)c2c(cccc2[N+](=O)[O-])/C1=N\c1ccccc1C",
        "CC(=O)c1c(C)n(CC2CCCO2)c2ccc(O)cc12",
    ]


@pytest.fixture
def smiles_failing_ghose() -> list[str]:
    return [
        r"CC(=O)C1=NN(c2ccccc2)C2(S1)S/C(=C\c1ccccc1)C(=O)N2c1ccccc1",
        "CC(=O)N(C)c1ccc(NC(=O)c2ccc(C(=O)Nc3ccc(N(C)C(C)=O)cc3)cc2)cc1",
        "CC(=O)n1cc(C2C=C(C(=O)N3CCN(Cc4ccc5c(c4)OCO5)CC3)OC(OCc3ccc(CO)cc3)C2)c2ccccc21",
    ]


@pytest.fixture
def smiles_passing_one_violation_ghose() -> list[str]:
    return [
        "CC(=O)c1c(C(C)=O)c(C)n(CCCCn2c(C)c(C(C)=O)c(C(C)=O)c2C)c1C",
        "CC(=O)c1c(N2CCN(Cc3ccc4c(c3)OCO4)CC2)nc(=S)n(-c2ccccc2)c1C",
        "CC(=O)Nc1ccc(NC(=O)CN(c2ccc(F)cc2)S(=O)(=O)c2ccc3c(c2)OCCO3)cc1",
    ]


def test_mols_passing_ghose(
    smiles_passing_ghose,
):
    mol_filter = GhoseFilter()
    smiles_filtered = mol_filter.transform(smiles_passing_ghose)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert len(smiles_filtered) == len(smiles_passing_ghose)


def test_mols_failing_ghose(smiles_failing_ghose):
    mol_filter = GhoseFilter()
    smiles_filtered = mol_filter.transform(smiles_failing_ghose)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert len(smiles_filtered) == 0


def test_mols_passing_with_violation_ghose(
    smiles_passing_one_violation_ghose,
):
    mol_filter = GhoseFilter(allow_one_violation=True)
    smiles_filtered = mol_filter.transform(smiles_passing_one_violation_ghose)
    assert len(smiles_filtered) == 3

    mol_filter = GhoseFilter(allow_one_violation=False)
    smiles_filtered = mol_filter.transform(smiles_passing_one_violation_ghose)
    assert len(smiles_filtered) == 0


def test_ghose_return_indicators(
    smiles_passing_ghose,
    smiles_failing_ghose,
    smiles_passing_one_violation_ghose,
):
    all_smiles = (
        smiles_passing_ghose + smiles_failing_ghose + smiles_passing_one_violation_ghose
    )

    mol_filter = GhoseFilter(return_indicators=True)
    filter_indicators = mol_filter.transform(all_smiles)
    expected_indicators = np.array(
        [True] * len(smiles_passing_ghose)
        + [False] * len(smiles_failing_ghose)
        + [False] * len(smiles_passing_one_violation_ghose),
        dtype=bool,
    )
    assert np.array_equal(filter_indicators, expected_indicators)

    mol_filter = GhoseFilter(allow_one_violation=True, return_indicators=True)
    filter_indicators = mol_filter.transform(all_smiles)
    expected_indicators = np.array(
        [True] * len(smiles_passing_ghose)
        + [False] * len(smiles_failing_ghose)
        + [True] * len(smiles_passing_one_violation_ghose),
        dtype=bool,
    )
    assert np.array_equal(filter_indicators, expected_indicators)


def test_ghose_parallel(smiles_list):
    mol_filter = GhoseFilter()
    mols_filtered_sequential = mol_filter.transform(smiles_list)

    mol_filter = GhoseFilter(n_jobs=-1, batch_size=1)
    mols_filtered_parallel = mol_filter.transform(smiles_list)

    assert mols_filtered_sequential == mols_filtered_parallel
