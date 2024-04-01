import numpy as np
from scipy.sparse import csr_array

from skfp.fingerprints import MAP4Fingerprint


def test_map4_bit_fingerprint(smallest_smiles_list, smallest_mols_list):
    map4_fp = MAP4Fingerprint(
        variant="bit",
        sparse=False,
        verbose=0,
        n_jobs=-1,
    )
    X_skfp = map4_fp.transform(smallest_smiles_list)

    X_map4 = np.stack(
        [map4_fp._calculate_single_mol_fingerprint(mol) for mol in smallest_mols_list],
        dtype=np.int64,
    )
    X_map4 = np.mod(X_map4, map4_fp.fp_size)
    X_map4 = np.stack([(np.bincount(x, minlength=map4_fp.fp_size) > 0) for x in X_map4])

    assert np.array_equal(X_skfp, X_map4)


def test_map4_count_fingerprint(smallest_smiles_list, smallest_mols_list):
    map4_fp = MAP4Fingerprint(
        variant="count",
        sparse=False,
        verbose=0,
        n_jobs=-1,
    )
    X_skfp = map4_fp.transform(smallest_smiles_list)

    X_map4 = np.stack(
        [map4_fp._calculate_single_mol_fingerprint(mol) for mol in smallest_mols_list],
        dtype=np.int64,
    )
    X_map4 = np.mod(X_map4, map4_fp.fp_size)
    X_map4 = np.stack([np.bincount(x, minlength=map4_fp.fp_size) for x in X_map4])

    assert np.array_equal(X_skfp, X_map4)


def test_map4_raw_hashes_fingerprint(smallest_smiles_list, smallest_mols_list):
    map4_fp = MAP4Fingerprint(
        variant="raw_hashes",
        sparse=False,
        verbose=0,
        n_jobs=-1,
    )
    X_skfp = map4_fp.transform(smallest_smiles_list)

    X_map4 = np.stack(
        [map4_fp._calculate_single_mol_fingerprint(mol) for mol in smallest_mols_list],
        dtype=np.int64,
    )

    assert np.array_equal(X_skfp, X_map4)


def test_map4_sparse_bit_fingerprint(smallest_smiles_list, smallest_mols_list):
    map4_fp = MAP4Fingerprint(
        variant="bit",
        sparse=True,
        verbose=0,
        n_jobs=-1,
    )
    X_skfp = map4_fp.transform(smallest_smiles_list)

    X_map4 = np.stack(
        [map4_fp._calculate_single_mol_fingerprint(mol) for mol in smallest_mols_list],
        dtype=np.int64,
    )
    X_map4 = np.mod(X_map4, map4_fp.fp_size)
    X_map4 = csr_array(
        [(np.bincount(x, minlength=map4_fp.fp_size) > 0) for x in X_map4],
        dtype=np.int64,
    )

    assert np.array_equal(X_skfp.data, X_map4.data)


def test_map4_sparse_count_fingerprint(smallest_smiles_list, smallest_mols_list):
    map4_fp = MAP4Fingerprint(
        variant="count",
        sparse=True,
        verbose=0,
        n_jobs=-1,
    )
    X_skfp = map4_fp.transform(smallest_smiles_list)

    X_map4 = np.stack(
        [map4_fp._calculate_single_mol_fingerprint(mol) for mol in smallest_mols_list],
        dtype=np.int64,
    )
    X_map4 = np.mod(X_map4, map4_fp.fp_size)
    X_map4 = csr_array(
        [np.bincount(x, minlength=map4_fp.fp_size) for x in X_map4], dtype=int
    )

    assert np.array_equal(X_skfp.data, X_map4.data)


def test_map4_sparse_raw_hashes_fingerprint(smallest_smiles_list, smallest_mols_list):
    map4_fp = MAP4Fingerprint(
        variant="raw_hashes",
        sparse=True,
        verbose=0,
        n_jobs=-1,
    )
    X_skfp = map4_fp.transform(smallest_smiles_list)

    X_map4 = csr_array(
        [map4_fp._calculate_single_mol_fingerprint(mol) for mol in smallest_mols_list]
    )

    assert np.array_equal(X_skfp.data, X_map4.data)
