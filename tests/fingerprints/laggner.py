import numpy as np
from scipy.sparse import csr_array

from skfp.fingerprints import LaggnerFingerprint


def test_laggner_bit_fingerprint(smiles_list):
    laggner_fp = LaggnerFingerprint(n_jobs=-1)
    X = laggner_fp.transform(smiles_list)

    assert isinstance(X, np.ndarray)
    assert X.dtype == np.uint8
    assert X.shape == (len(smiles_list), 307)
    assert np.all(np.isin(X, [0, 1]))


def test_laggner_count_fingerprint(smiles_list):
    laggner_fp = LaggnerFingerprint(count=True, n_jobs=-1)
    X = laggner_fp.transform(smiles_list)

    assert isinstance(X, np.ndarray)
    assert X.dtype == np.uint32
    assert X.shape == (len(smiles_list), 307)
    assert np.all(X >= 0)


def test_laggner_bit_sparse_fingerprint(smiles_list):
    laggner_fp = LaggnerFingerprint(sparse=True, n_jobs=-1)
    X = laggner_fp.transform(smiles_list)

    assert isinstance(X, csr_array)
    assert X.dtype == np.uint8
    assert X.shape == (len(smiles_list), 307)
    assert np.all(X.data == 1)


def test_laggner_count_sparse_fingerprint(smiles_list):
    laggner_fp = LaggnerFingerprint(sparse=True, count=True, n_jobs=-1)
    X = laggner_fp.transform(smiles_list)

    assert isinstance(X, csr_array)
    assert X.dtype == np.uint32
    assert X.shape == (len(smiles_list), 307)
    assert np.all(X.data > 0)


def test_laggner_salt_matching():
    non_salts = ["O", "CC", "[C-]#N", "CC=O"]
    salts = ["[Na+].[Cl-]", "[Ca+2].[Cl-].[Cl-]"]
    smiles_list = non_salts + salts

    laggner_fp = LaggnerFingerprint()
    X = laggner_fp.transform(smiles_list)

    assert X[0, 298] == 0
    assert X[1, 298] == 0
    assert X[2, 298] == 0
    assert X[3, 298] == 0
    assert X[4, 298] == 1
    assert X[5, 298] == 1


def test_laggner_feature_names():
    # we check a few selected feature names
    laggner_fp = LaggnerFingerprint()
    feature_names = laggner_fp.get_feature_names_out()

    assert len(feature_names) == laggner_fp.n_features_out

    assert feature_names[0] == "Primary_carbon"
    assert feature_names[1] == "Secondary_carbon"
    assert feature_names[2] == "Tertiary_carbon"
    assert feature_names[3] == "Quaternary_carbon"

    assert feature_names[4] == "Alkene"
    assert feature_names[5] == "Alkyne"
    assert feature_names[6] == "Allene"

    assert feature_names[-4] == "Dicarbodiazene"
    assert feature_names[-3] == "CH-acidic"
    assert feature_names[-2] == "CH-acidic_strong"
    assert feature_names[-1] == "Chiral_center_specified"
