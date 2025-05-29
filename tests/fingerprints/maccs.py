import numpy as np
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdMolDescriptors import GetMACCSKeysFingerprint
from scipy.sparse import csr_array

from skfp.fingerprints import MACCSFingerprint


def test_maccs_bit_fingerprint(smiles_list, mols_list):
    maccs_fp = MACCSFingerprint(n_jobs=-1)
    X_skfp = maccs_fp.transform(smiles_list)
    X_rdkit = np.array([GetMACCSKeysFingerprint(mol) for mol in mols_list])

    assert np.array_equal(X_skfp, X_rdkit[:, 1:])  # ignore first, all-zeros column
    assert X_skfp.shape == (len(smiles_list), 166)
    assert X_skfp.dtype == np.uint8
    assert np.all(np.isin(X_skfp, [0, 1]))


def test_maccs_count_fingerprint(smiles_list, mols_list):
    maccs_fp = MACCSFingerprint(count=True, n_jobs=-1)
    X_skfp = maccs_fp.transform(smiles_list)

    assert X_skfp.shape == (len(smiles_list), 157)
    assert X_skfp.dtype == np.uint32
    assert np.all(X_skfp >= 0)


def test_maccs_sparse_bit_fingerprint(smiles_list, mols_list):
    maccs_fp = MACCSFingerprint(sparse=True, n_jobs=-1)
    X_skfp = maccs_fp.transform(smiles_list)
    X_rdkit = csr_array([GetMACCSKeysFingerprint(mol) for mol in mols_list])

    assert np.array_equal(X_skfp.data, X_rdkit.data)
    assert X_skfp.shape == (len(smiles_list), 166)
    assert X_skfp.dtype == np.uint8
    assert np.all(X_skfp.data == 1)


def test_maccs_sparse_count_fingerprint(smiles_list, mols_list):
    maccs_fp = MACCSFingerprint(count=True, sparse=True, n_jobs=-1)
    X_skfp = maccs_fp.transform(smiles_list)

    assert X_skfp.shape == (len(smiles_list), 157)
    assert X_skfp.dtype == np.uint32
    assert np.all(X_skfp.data >= 0)


def test_maccs_feature_names():
    # we check a few selected feature names
    maccs_fp = MACCSFingerprint()
    feature_names = maccs_fp.get_feature_names_out()

    assert len(feature_names) == maccs_fp.n_features_out

    assert feature_names[0] == "ISOTOPE"
    assert feature_names[1] == "atomic num >103"
    assert feature_names[2] == "Group IVa,Va,VIa Rows 4-6 "

    assert feature_names[-4] == "6M Ring"
    assert feature_names[-3] == "O"
    assert feature_names[-2] == "Ring"
    assert feature_names[-1] == "Fragments"


def test_maccs_count_feature_names():
    # we check a few selected feature names
    maccs_fp = MACCSFingerprint(count=True)
    feature_names = maccs_fp.get_feature_names_out()

    assert len(feature_names) == maccs_fp.n_features_out

    assert feature_names[0] == "fragments"
    assert feature_names[1] == "N"
    assert feature_names[2] == "O"
    assert feature_names[3] == "F"

    assert feature_names[-3] == "QCH2A"
    assert feature_names[-2] == "A!CH2!A"
    assert feature_names[-1] == "NA(A)A"


def test_maccs_patterns():
    maccs_fp = MACCSFingerprint()
    mol = MolFromSmiles("[Na+].[Cl-]")
    counts = maccs_fp._get_maccs_patterns_counts(mol)
    assert len(counts) == 157

    assert counts[0] == 2  # fragments
    assert counts[1] == 0  # N
    assert counts[7] == 1  # Cl
    assert counts[15] == 1  # Na
