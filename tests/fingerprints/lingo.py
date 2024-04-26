import os

import numpy as np
from scipy.sparse import load_npz

from skfp.fingerprints import LingoFingerprint


def test_lingo_fingerprint_smiles_to_dict():
    smiles = ["CC(=O)NCCC1=CNC2=C1C=C(C=C2)OC"]
    lingo_fp = LingoFingerprint()
    X_skfp = lingo_fp.smiles_to_dicts(smiles)
    expected = [
        {
            "CC(=": 1,
            "C(=O": 1,
            "(=O)": 1,
            "=O)N": 1,
            "O)NC": 1,
            ")NCC": 1,
            "NCCC": 1,
            "CCC0": 1,
            "CC0=": 1,
            "C0=C": 2,
            "0=CN": 1,
            "=CNC": 1,
            "CNC0": 1,
            "NC0=": 1,
            "0=C0": 1,
            "=C0C": 1,
            "C0C=": 1,
            "0C=C": 1,
            "C=C(": 1,
            "=C(C": 1,
            "C(C=": 1,
            "(C=C": 1,
            "C=C0": 1,
            "=C0)": 1,
            "C0)O": 1,
            "0)OC": 1,
        }
    ]
    assert X_skfp == expected


def test_lingo_fingerprint_bit():
    smiles = ["CC(=O)NCCC1=CNC2=C1C=C(C=C2)OC", "C[n]1cnc2N(C)C(=O)N(C)C(=O)c12"]
    lingo_fp = LingoFingerprint()
    X_skfp = lingo_fp.transform(smiles)

    expected_array = np.load(
        os.path.join("tests", "fingerprints", "data", "lingo_bit_fp.npy")
    )

    assert np.array_equal(X_skfp, expected_array)
    assert X_skfp.shape == (2, 1024)
    assert X_skfp.dtype == np.uint8
    assert np.all(np.isin(X_skfp, [0, 1]))


def test_lingo_fingerprint_count():
    smiles = ["CC(=O)NCCC1=CNC2=C1C=C(C=C2)OC", "C[n]1cnc2N(C)C(=O)N(C)C(=O)c12"]
    lingo_fp = LingoFingerprint(count=True)
    X_skfp = lingo_fp.transform(smiles)

    expected_array = np.load(
        os.path.join("tests", "fingerprints", "data", "lingo_count_fp.npy")
    )

    assert np.array_equal(X_skfp, expected_array)
    assert X_skfp.shape == (2, 1024)
    assert X_skfp.dtype == np.uint32
    assert np.all(X_skfp >= 0)


def test_lingo_fingerprint_bit_sparse():
    smiles = ["CC(=O)NCCC1=CNC2=C1C=C(C=C2)OC", "C[n]1cnc2N(C)C(=O)N(C)C(=O)c12"]
    lingo_fp = LingoFingerprint(sparse=True)
    X_skfp = lingo_fp.transform(smiles)

    expected_array = load_npz(
        os.path.join("tests", "fingerprints", "data", "lingo_bit_sparse.npz")
    )

    assert np.array_equal(X_skfp.data, expected_array.data)
    assert X_skfp.shape == (2, 1024)
    assert X_skfp.dtype == np.uint8
    assert np.all(X_skfp.data == 1)


def test_lingo_fingerprint_count_sparse():
    smiles = ["CC(=O)NCCC1=CNC2=C1C=C(C=C2)OC", "C[n]1cnc2N(C)C(=O)N(C)C(=O)c12"]
    lingo_fp = LingoFingerprint(count=True, sparse=True)
    X_skfp = lingo_fp.transform(smiles)

    expected_array = load_npz(
        os.path.join("tests", "fingerprints", "data", "lingo_count_sparse.npz")
    )

    assert np.array_equal(X_skfp.data, expected_array.data)
    assert X_skfp.shape == (2, 1024)
    assert X_skfp.dtype == np.uint32
    assert np.all(X_skfp.data > 0)
