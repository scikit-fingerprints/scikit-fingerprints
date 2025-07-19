import numpy as np
from rdkit.Chem import Mol

from skfp.filters import ZINCDruglikeFilter


def test_zinc_druglike(mols_list):
    pains = ZINCDruglikeFilter()
    mols_filtered = pains.transform(mols_list)
    assert all(isinstance(x, Mol) for x in mols_filtered)
    assert len(mols_filtered) <= len(mols_list)


def test_zinc_druglike_parallel(smiles_list):
    filt = ZINCDruglikeFilter()
    smiles_filtered_sequential = filt.transform(smiles_list)

    filt = ZINCDruglikeFilter(n_jobs=-1)
    smiles_filtered_parallel = filt.transform(smiles_list)

    assert smiles_filtered_sequential == smiles_filtered_parallel


def test_zinc_druglike_allowing_one_violation(mols_list):
    filt = ZINCDruglikeFilter()
    filt_loose = ZINCDruglikeFilter(allow_one_violation=True)

    mols_filtered = filt.transform(mols_list)
    mols_filtered_loose = filt_loose.transform(mols_list)

    assert len(mols_filtered) <= len(mols_filtered_loose)
    assert len(mols_filtered_loose) <= len(mols_list)


def test_zinc_druglike_transform_x_y(mols_list):
    labels = np.ones(len(mols_list))

    filt = ZINCDruglikeFilter()
    mols_filtered, labels_filt = filt.transform_x_y(mols_list, labels)
    assert len(mols_filtered) == len(labels_filt)


def test_zinc_druglike_condition_names():
    filt = ZINCDruglikeFilter()
    condition_names = filt.get_feature_names_out()

    assert isinstance(condition_names, np.ndarray)
    assert condition_names.shape == (13,)


def test_zinc_druglike_return_condition_indicators(mols_list):
    filt = ZINCDruglikeFilter(return_type="condition_indicators")
    condition_indicators = filt.transform(mols_list)

    assert isinstance(condition_indicators, np.ndarray)
    assert condition_indicators.shape == (len(mols_list), 13)
    assert np.issubdtype(condition_indicators.dtype, bool)
    assert np.all(np.isin(condition_indicators, [0, 1]))


def test_zinc_druglike_return_condition_indicators_transform_x_y(mols_list):
    labels = np.ones(len(mols_list))

    filt = ZINCDruglikeFilter(return_type="condition_indicators")
    condition_indicators, y = filt.transform_x_y(mols_list, labels)

    assert isinstance(condition_indicators, np.ndarray)
    assert condition_indicators.shape == (len(mols_list), 13)
    assert np.issubdtype(condition_indicators.dtype, bool)
    assert np.all(np.isin(condition_indicators, [0, 1]))
    assert len(condition_indicators) == len(y)
