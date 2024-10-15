import numpy as np
import pytest

from skfp.filters import TiceHerbicidesFilter


@pytest.fixture
def smiles_passing_tice_herbicides() -> list[str]:
    return [
        "CCC(C)NC(=O)NC(CCSC)C(=O)OC",
        "OCCNc1nc2ccc(Cl)cc2[nH]1",
        "Nc1nnc(-c2ccco2)s1",
    ]


@pytest.fixture
def smiles_failing_tice_herbicides() -> list[str]:
    return [
        "C(=Cc1ccccc1)C1=[O+][Cu-3]2([O+]=C(C=Cc3ccccc3)CC(c3ccccc3)=[O+]2)[O+]=C(c2ccccc2)C1",
        "CS(C)=O",
        "O=[As]O",
    ]


@pytest.fixture
def smiles_passing_one_violation_tice_herbicides() -> list[str]:
    return [
        "CC(C)c1c(C#N)c(N)nc(SCC(=O)c2cc3ccccc3oc2=O)c1C#N",
        "Cc1ccc(NC(=O)N(Cc2ccc3c(c2)OCO3)Cc2cc3cc4c(cc3[nH]c2=O)OCCO4)cc1",
        "S=C1NCCS1",
    ]


def test_mols_passing_tice_herbicides(
    smiles_passing_tice_herbicides,
):
    mol_filter = TiceHerbicidesFilter()
    smiles_filtered = mol_filter.transform(smiles_passing_tice_herbicides)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert len(smiles_filtered) == len(smiles_passing_tice_herbicides)


def test_mols_failing_tice_herbicides(smiles_failing_tice_herbicides):
    mol_filter = TiceHerbicidesFilter()
    smiles_filtered = mol_filter.transform(smiles_failing_tice_herbicides)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert len(smiles_filtered) == 0


def test_mols_passing_with_violation_tice_herbicides(
    smiles_passing_one_violation_tice_herbicides,
):
    mol_filter = TiceHerbicidesFilter(allow_one_violation=True)
    smiles_filtered = mol_filter.transform(smiles_passing_one_violation_tice_herbicides)
    assert len(smiles_filtered) == 3

    mol_filter = TiceHerbicidesFilter(allow_one_violation=False)
    smiles_filtered = mol_filter.transform(smiles_passing_one_violation_tice_herbicides)
    assert len(smiles_filtered) == 0


def test_tice_herbicides_return_indicators(
    smiles_passing_tice_herbicides,
    smiles_failing_tice_herbicides,
    smiles_passing_one_violation_tice_herbicides,
):
    all_smiles = (
        smiles_passing_tice_herbicides
        + smiles_failing_tice_herbicides
        + smiles_passing_one_violation_tice_herbicides
    )

    mol_filter = TiceHerbicidesFilter(return_indicators=True)
    filter_indicators = mol_filter.transform(all_smiles)
    expected_indicators = np.array(
        [True] * len(smiles_passing_tice_herbicides)
        + [False] * len(smiles_failing_tice_herbicides)
        + [False] * len(smiles_passing_one_violation_tice_herbicides),
        dtype=bool,
    )
    assert np.array_equal(filter_indicators, expected_indicators)

    mol_filter = TiceHerbicidesFilter(allow_one_violation=True, return_indicators=True)
    filter_indicators = mol_filter.transform(all_smiles)
    expected_indicators = np.array(
        [True] * len(smiles_passing_tice_herbicides)
        + [False] * len(smiles_failing_tice_herbicides)
        + [True] * len(smiles_passing_one_violation_tice_herbicides),
        dtype=bool,
    )
    assert np.array_equal(filter_indicators, expected_indicators)


def test_tice_herbicides_parallel(smiles_list):
    mol_filter = TiceHerbicidesFilter()
    mols_filtered_sequential = mol_filter.transform(smiles_list)

    mol_filter = TiceHerbicidesFilter(n_jobs=-1, batch_size=1)
    mols_filtered_parallel = mol_filter.transform(smiles_list)

    assert mols_filtered_sequential == mols_filtered_parallel
