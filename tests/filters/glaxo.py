import numpy as np
from rdkit.Chem import Mol

from skfp.filters import GlaxoFilter


def test_glaxo(mols_list):
    pains = GlaxoFilter()
    mols_filtered = pains.transform(mols_list)
    assert all(isinstance(x, Mol) for x in mols_filtered)
    assert len(mols_filtered) <= len(mols_list)


def test_glaxo_parallel(smiles_list):
    filt = GlaxoFilter()
    smiles_filtered_sequential = filt.transform(smiles_list)

    filt = GlaxoFilter(n_jobs=-1)
    smiles_filtered_parallel = filt.transform(smiles_list)

    assert smiles_filtered_sequential == smiles_filtered_parallel


def test_glaxo_allowing_one_violation(mols_list):
    filt = GlaxoFilter()
    filt_loose = GlaxoFilter(allow_one_violation=True)

    mols_filtered = filt.transform(mols_list)
    mols_filtered_loose = filt_loose.transform(mols_list)

    assert len(mols_filtered) <= len(mols_filtered_loose)
    assert len(mols_filtered_loose) <= len(mols_list)


def test_glaxo_transform_x_y(mols_list):
    labels = np.ones(len(mols_list))

    filt = GlaxoFilter()
    mols_filtered, labels_filt = filt.transform_x_y(mols_list, labels)
    assert len(mols_filtered) == len(labels_filt)
