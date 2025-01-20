from inspect import getmembers, isfunction

import numpy as np
import rdkit.Chem.Fragments
from scipy.sparse import csr_array

from skfp.fingerprints import FunctionalGroupsFingerprint


def test_functional_groups_bit_fingerprint(smiles_list, mols_list):
    fg_fp = FunctionalGroupsFingerprint(n_jobs=-1)
    X_skfp = fg_fp.transform(smiles_list)

    func_groups_functions = [
        function
        for name, function in getmembers(rdkit.Chem.Fragments, isfunction)
        if name.startswith("fr_")
    ]
    X_rdkit = [
        [func_group(mol) > 0 for func_group in func_groups_functions]
        for mol in mols_list
    ]
    X_rdkit = np.array(X_rdkit, dtype=np.uint8)

    assert np.array_equal(X_skfp, X_rdkit)
    assert X_skfp.shape == (len(smiles_list), 85)
    assert X_skfp.dtype == np.uint8
    assert np.all(np.isin(X_skfp, [0, 1]))


def test_functional_groups_count_fingerprint(smiles_list, mols_list):
    fg_fp = FunctionalGroupsFingerprint(count=True, n_jobs=-1)
    X_skfp = fg_fp.transform(smiles_list)

    func_groups_functions = [
        function
        for name, function in getmembers(rdkit.Chem.Fragments, isfunction)
        if name.startswith("fr_")
    ]
    X_rdkit = [
        [func_group(mol) for func_group in func_groups_functions] for mol in mols_list
    ]
    X_rdkit = np.array(X_rdkit, dtype=np.uint32)

    assert np.array_equal(X_skfp, X_rdkit)
    assert X_skfp.shape == (len(smiles_list), 85)
    assert X_skfp.dtype == np.uint32
    assert np.all(X_skfp >= 0)


def test_functional_groups_sparse_bit_fingerprint(smiles_list, mols_list):
    fg_fp = FunctionalGroupsFingerprint(sparse=True, n_jobs=-1)
    X_skfp = fg_fp.transform(smiles_list)

    func_groups_functions = [
        function
        for name, function in getmembers(rdkit.Chem.Fragments, isfunction)
        if name.startswith("fr_")
    ]
    X_rdkit = [
        [func_group(mol) > 0 for func_group in func_groups_functions]
        for mol in mols_list
    ]
    X_rdkit = csr_array(X_rdkit, dtype=np.uint8)

    assert np.array_equal(X_skfp.data, X_rdkit.data)
    assert X_skfp.shape == (len(smiles_list), 85)
    assert X_skfp.dtype == np.uint8
    assert np.all(X_skfp.data == 1)


def test_functional_groups_sparse_count_fingerprint(smiles_list, mols_list):
    fg_fp = FunctionalGroupsFingerprint(count=True, sparse=True, n_jobs=-1)
    X_skfp = fg_fp.transform(smiles_list)

    func_groups_functions = [
        function
        for name, function in getmembers(rdkit.Chem.Fragments, isfunction)
        if name.startswith("fr_")
    ]
    X_rdkit = [
        [func_group(mol) for func_group in func_groups_functions] for mol in mols_list
    ]
    X_rdkit = csr_array(X_rdkit, dtype=np.uint32)

    assert np.array_equal(X_skfp.data, X_rdkit.data)
    assert X_skfp.shape == (len(smiles_list), 85)
    assert X_skfp.dtype == np.uint32
    assert np.all(X_skfp.data > 0)


def test_functional_groups_feature_names():
    fg_fp = FunctionalGroupsFingerprint()
    feature_names = fg_fp.get_feature_names_out()
    assert len(feature_names) == fg_fp.n_features_out

    # compared with https://rdkit.org/docs/source/rdkit.Chem.Fragments.html
    assert feature_names[0] == "aliphatic carboxylic acids"
    assert feature_names[1] == "aliphatic hydroxyl groups"
    assert (
        feature_names[-2]
        == "unbranched alkanes of at least 4 members (excludes halogenated alkanes)"
    )
    assert feature_names[-1] == "urea groups"
