from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from scipy.sparse import csr_array

from skfp.fingerprints import KlekotaRothFingerprint


def test_klekota_roth_bit_fingerprint(smiles_list):
    kr_fp = KlekotaRothFingerprint(n_jobs=-1)
    X = kr_fp.transform(smiles_list)

    assert isinstance(X, np.ndarray)
    assert X.dtype == np.uint8
    assert X.shape == (len(smiles_list), 4860)
    assert np.all(np.isin(X, [0, 1]))


def test_klekota_roth_count_fingerprint(smiles_list):
    kr_fp = KlekotaRothFingerprint(count=True, n_jobs=-1)
    X = kr_fp.transform(smiles_list)

    assert isinstance(X, np.ndarray)
    assert X.dtype == np.uint32
    assert X.shape == (len(smiles_list), 4860)
    assert np.all(X >= 0)


def test_klekota_roth_bit_sparse_fingerprint(smiles_list):
    kr_fp = KlekotaRothFingerprint(sparse=True, n_jobs=-1)
    X = kr_fp.transform(smiles_list)

    assert isinstance(X, csr_array)
    assert X.dtype == np.uint8
    assert X.shape == (len(smiles_list), 4860)
    assert np.all(X.data == 1)


def test_klekota_roth_count_sparse_fingerprint(smiles_list):
    kr_fp = KlekotaRothFingerprint(sparse=True, count=True, n_jobs=-1)
    X = kr_fp.transform(smiles_list)

    assert isinstance(X, csr_array)
    assert X.dtype == np.uint32
    assert X.shape == (len(smiles_list), 4860)
    assert np.all(X.data > 0)


def test_klekota_roth_feature_names():
    kr_fp = KlekotaRothFingerprint()
    feature_names = kr_fp.get_feature_names_out()

    assert len(feature_names) == kr_fp.n_features_out
    assert len(feature_names) == len(set(feature_names))


@pytest.mark.parametrize("count", [False, True])
@pytest.mark.parametrize("n_jobs", [-1, 1])
def test_klekota_roth_result_correctness(smiles_list, count, n_jobs):
    patterns_path = Path(__file__).parent / "data" / "klekota_roth_patterns.txt"
    with patterns_path.open(encoding="utf-8") as f:
        patterns = [line.strip() for line in f if line.strip()]

    compiled_patterns = [Chem.MolFromSmarts(p) for p in patterns]

    kr_fp = KlekotaRothFingerprint(count=count, n_jobs=n_jobs)
    X = kr_fp.transform(smiles_list)

    n_features = kr_fp.n_features_out

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        expected = np.zeros(n_features, dtype=X.dtype)

        for j, patt in enumerate(compiled_patterns):
            if count:
                n_matches = len(mol.GetSubstructMatches(patt))
                expected[j] = n_matches
            elif mol.HasSubstructMatch(patt):
                expected[j] = 1
        assert np.array_equal(X[i], expected)
