import numpy as np
from scipy.sparse import csr_array

from skfp.fingerprints import MAPFingerprint


def test_map_bit_fingerprint(smallest_smiles_list, smallest_mols_list):
    map_fp = MAPFingerprint(
        variant="bit",
        sparse=False,
        verbose=0,
        n_jobs=-1,
    )
    X_skfp = map_fp.transform(smallest_smiles_list)

    X_map = np.stack(
        [
            map_fp._calculate_single_mol_fingerprint(mol) > 0
            for mol in smallest_mols_list
        ],
    )

    assert np.array_equal(X_skfp, X_map)
    assert X_skfp.shape == (len(smallest_smiles_list), map_fp.fp_size)


def test_map_count_fingerprint(smallest_smiles_list, smallest_mols_list):
    map_fp = MAPFingerprint(
        variant="count",
        sparse=False,
        verbose=0,
        n_jobs=-1,
    )
    X_skfp = map_fp.transform(smallest_smiles_list)

    X_map = np.stack(
        [map_fp._calculate_single_mol_fingerprint(mol) for mol in smallest_mols_list],
    )

    assert np.array_equal(X_skfp, X_map)
    assert X_skfp.shape == (len(smallest_smiles_list), map_fp.fp_size)


def test_map_raw_hashes_fingerprint(smallest_smiles_list, smallest_mols_list):
    map_fp = MAPFingerprint(
        variant="raw_hashes",
        sparse=False,
        verbose=0,
        n_jobs=-1,
    )
    X_skfp = map_fp.transform(smallest_smiles_list)

    X_map = np.stack(
        [map_fp._calculate_single_mol_fingerprint(mol) for mol in smallest_mols_list]
    )

    assert np.array_equal(X_skfp, X_map)
    assert X_skfp.shape == (len(smallest_smiles_list), map_fp.fp_size)


def test_map_sparse_bit_fingerprint(smallest_smiles_list, smallest_mols_list):
    map_fp = MAPFingerprint(
        variant="bit",
        sparse=True,
        verbose=0,
        n_jobs=-1,
    )
    X_skfp = map_fp.transform(smallest_smiles_list)

    X_map = csr_array(
        [
            map_fp._calculate_single_mol_fingerprint(mol) > 0
            for mol in smallest_mols_list
        ],
    )

    assert np.array_equal(X_skfp.data, X_map.data)
    assert X_skfp.shape == (len(smallest_smiles_list), map_fp.fp_size)


def test_map_sparse_count_fingerprint(smallest_smiles_list, smallest_mols_list):
    map_fp = MAPFingerprint(
        variant="count",
        sparse=True,
        verbose=0,
        n_jobs=-1,
    )
    X_skfp = map_fp.transform(smallest_smiles_list)

    X_map = csr_array(
        [map_fp._calculate_single_mol_fingerprint(mol) for mol in smallest_mols_list],
    )

    assert np.array_equal(X_skfp.data, X_map.data)
    assert X_skfp.shape == (len(smallest_smiles_list), map_fp.fp_size)


def test_map_sparse_raw_hashes_fingerprint(smallest_smiles_list, smallest_mols_list):
    map_fp = MAPFingerprint(
        variant="raw_hashes",
        sparse=True,
        verbose=0,
        n_jobs=-1,
    )
    X_skfp = map_fp.transform(smallest_smiles_list)

    X_map = csr_array(
        [map_fp._calculate_single_mol_fingerprint(mol) for mol in smallest_mols_list]
    )

    assert np.array_equal(X_skfp.data, X_map.data)
    assert X_skfp.shape == (len(smallest_smiles_list), map_fp.fp_size)
