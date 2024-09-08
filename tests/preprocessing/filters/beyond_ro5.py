import numpy as np
import pytest
from rdkit.Chem import Mol

from skfp.preprocessing import BeyondRO5Filter, LipinskiFilter


@pytest.fixture
def smiles_passing_ro5() -> list[str]:
    return [
        # paracetamol
        "CC(=O)Nc1ccc(O)cc1",
        # caffeine
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        # nicotine
        "c1ncccc1[C@@H]2CCCN2C",
    ]


@pytest.fixture
def smiles_failing_ro5() -> list[str]:
    return [
        # Telaprevir
        "CCC[C@@H](C(=O)C(=O)NC1CC1)NC(=O)[C@@H]2[C@H]3CCC[C@H]"
        "3CN2C(=O)[C@H](C(C)(C)C)NC(=O)[C@H](C4CCCCC4)NC(=O)c5cnccn5",
        # Atorvastatin
        "O=C(O)C[C@H](O)C[C@H](O)CCn2c(c(c(c2c1ccc(F)cc1)c3ccccc3)C(=O)Nc4ccccc4)C(C)C",
        # Telmisartan
        "O=C(O)c1ccccc1c2ccc(cc2)Cn3c4cc(cc(c4nc3CCC)C)c5nc6ccccc6n5C",
    ]


@pytest.fixture
def smiles_beyond_ro5() -> list[str]:
    return [
        # Lapatinib
        "CS(=O)(=O)CCNCc1ccc(o1)c2ccc3c(c2)c(ncn3)Nc4ccc(c(c4)Cl)OCc5cccc(c5)F",
        # Aliskiren
        "O=C(N)C(C)(C)CNC(=O)[C@H](C(C)C)C[C@H](O)[C@@H](N)C[C@@H](C(C)C)Cc1cc(OCCCOC)c(OC)cc1",
        # Ergotamine
        "C=12CCC=3C=C(C=C(C3[C@H](C1N=CC(=C2)Br)C4CCN(CC4)C(=O)CC5CCN(CC5)C(N)=O)Br)Cl",
    ]


def test_mols_passing_bro5(smiles_passing_ro5, smiles_failing_ro5, smiles_beyond_ro5):
    filt = BeyondRO5Filter()

    smiles_filtered = filt.transform(smiles_passing_ro5)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert len(smiles_filtered) == len(smiles_passing_ro5)

    smiles_filtered = filt.transform(smiles_failing_ro5)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert len(smiles_filtered) == len(smiles_failing_ro5)

    smiles_filtered = filt.transform(smiles_beyond_ro5)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert len(smiles_filtered) == len(smiles_beyond_ro5)


def test_mols_ro5_vs_bro5(smiles_passing_ro5, smiles_failing_ro5, smiles_beyond_ro5):
    all_smiles = smiles_passing_ro5 + smiles_failing_ro5 + smiles_beyond_ro5

    filt_ro5 = LipinskiFilter()
    assert len(filt_ro5.transform(smiles_passing_ro5)) == 3
    assert len(filt_ro5.transform(smiles_failing_ro5)) == 0
    assert len(filt_ro5.transform(smiles_beyond_ro5)) == 0
    assert len(filt_ro5.transform(all_smiles)) == 3

    filt_bro5 = BeyondRO5Filter()
    assert len(filt_bro5.transform(smiles_passing_ro5)) == 3
    assert len(filt_bro5.transform(smiles_failing_ro5)) == 3
    assert len(filt_bro5.transform(smiles_beyond_ro5)) == 3
    assert len(filt_bro5.transform(all_smiles)) == 9


def test_bro5_smiles_and_mol_input(smiles_list, mols_list):
    filt = BeyondRO5Filter()
    smiles_filtered = filt.transform(smiles_list)
    mols_filtered = filt.transform(mols_list)

    assert all(isinstance(x, str) for x in smiles_filtered)
    assert all(isinstance(x, Mol) for x in mols_filtered)
    assert len(smiles_filtered) == len(mols_filtered)


def test_bro5_parallel(smiles_list):
    filt = BeyondRO5Filter()
    mols_filtered_sequential = filt.transform(smiles_list)

    filt = BeyondRO5Filter(n_jobs=-1, batch_size=1)
    mols_filtered_parallel = filt.transform(smiles_list)

    assert mols_filtered_sequential == mols_filtered_parallel


def test_mols_loose_bro5(mols_list):
    filt = BeyondRO5Filter()
    mols_filtered_bro5 = filt.transform(mols_list)

    filt = BeyondRO5Filter(allow_one_violation=True)
    mols_filtered_loose_bro5 = filt.transform(mols_list)

    assert len(mols_filtered_bro5) <= len(mols_filtered_loose_bro5)


def test_bro5_return_indicators(
    smiles_passing_ro5, smiles_failing_ro5, smiles_beyond_ro5
):
    all_smiles = smiles_passing_ro5 + smiles_failing_ro5 + smiles_beyond_ro5

    filt = BeyondRO5Filter(return_indicators=True)
    filter_indicators = filt.transform(all_smiles)
    assert np.all(filter_indicators)
