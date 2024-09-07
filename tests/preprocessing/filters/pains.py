import pytest
from rdkit.Chem import Mol

from skfp.preprocessing import PAINSFilter


def test_basic_pains(mols_list):
    pains = PAINSFilter(n_jobs=-1)
    mols_filtered = pains.transform(mols_list)
    assert all(isinstance(x, Mol) for x in mols_filtered)
    assert len(mols_filtered) <= len(mols_list)


def test_pains_variants(mols_list):
    pains_a = PAINSFilter(variant="A", n_jobs=-1)
    pains_b = PAINSFilter(variant="B", n_jobs=-1)
    pains_c = PAINSFilter(variant="C", n_jobs=-1)

    mols_filtered_a = pains_a.transform(mols_list)
    mols_filtered_b = pains_b.transform(mols_list)
    mols_filtered_c = pains_c.transform(mols_list)

    assert len(mols_filtered_a) <= len(mols_filtered_b)
    assert len(mols_filtered_b) <= len(mols_filtered_c)


def test_pains_allowing_one_violation(mols_list):
    pains = PAINSFilter(variant="C", n_jobs=-1)
    pains_loose = PAINSFilter(variant="C", allow_one_violation=True, n_jobs=-1)

    mols_filtered = pains.transform(mols_list)
    mols_filtered_loose = pains_loose.transform(mols_list)

    assert len(mols_filtered) <= len(mols_filtered_loose)
    assert len(mols_filtered_loose) <= len(mols_list)


def test_pains_wrong_variant():
    with pytest.raises(ValueError) as exc_info:
        PAINSFilter(variant="D")

    assert 'PAINS variant must be "A", "B" or "C", got' in str(exc_info)
