import numpy as np
import scipy.sparse

from skfp.fingerprints import MAP4Fingerprint


def test_map4_bit_fingerprint(smiles_list, mols_list):
    map4_fp = MAP4Fingerprint(
        count=False,
        sparse=False,
        verbose=0,
        n_jobs=-1,
    )
    X_skfp = map4_fp.transform(smiles_list)

    X_map4 = np.stack(
        [
            map4_fp._calculate_single_mol_fingerprint(mol, count=False)
            for mol in mols_list
        ]
    )

    assert np.array_equal(X_skfp, X_map4)


def test_map4_count_fingerprint(smiles_list, mols_list):
    map4_fp = MAP4Fingerprint(
        count=True,
        sparse=False,
        verbose=0,
        n_jobs=-1,
    )
    X_skfp = map4_fp.transform(smiles_list)

    X_map4 = np.stack(
        [
            map4_fp._calculate_single_mol_fingerprint(mol, count=True)
            for mol in mols_list
        ]
    )

    assert np.array_equal(X_skfp, X_map4)


def test_map4_sparse_bit_fingerprint(smiles_list, mols_list):
    map4_fp = MAP4Fingerprint(
        count=False,
        sparse=True,
        verbose=0,
        n_jobs=-1,
    )
    X_skfp = map4_fp.transform(smiles_list)

    X_map4 = scipy.sparse.vstack(
        [
            map4_fp._calculate_single_mol_fingerprint(mol, count=False)
            for mol in mols_list
        ]
    )

    assert np.array_equal(X_skfp.data, X_map4.data)


def test_map4_sparse_count_fingerprint(smiles_list, mols_list):
    map4_fp = MAP4Fingerprint(
        count=True,
        sparse=True,
        verbose=0,
        n_jobs=-1,
    )
    X_skfp = map4_fp.transform(smiles_list)

    X_map4 = scipy.sparse.vstack(
        [
            map4_fp._calculate_single_mol_fingerprint(mol, count=True)
            for mol in mols_list
        ]
    )

    assert np.array_equal(X_skfp.data, X_map4.data)
