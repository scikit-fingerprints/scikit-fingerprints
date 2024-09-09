import numpy as np
from rdkit.Chem.EState.EState_VSA import EState_VSA_
from rdkit.Chem.rdMolDescriptors import PEOE_VSA_, SMR_VSA_, SlogP_VSA_
from scipy.sparse import csr_array

from skfp.fingerprints import VSAFingerprint


def test_vsa_fingerprint(mols_list):
    vsa_fp = VSAFingerprint(n_jobs=-1)
    X_skfp = vsa_fp.transform(mols_list)

    X_slogp = np.array([SlogP_VSA_(mol) for mol in mols_list])
    X_smr = np.array([SMR_VSA_(mol) for mol in mols_list])
    X_peoe = np.array([PEOE_VSA_(mol) for mol in mols_list])

    X_rdkit = np.column_stack((X_slogp, X_smr, X_peoe))

    assert np.allclose(X_skfp, X_rdkit)
    assert X_skfp.shape == (len(mols_list), 36)
    assert np.issubdtype(X_skfp.dtype, np.floating)


def test_vsa_fingerprint_with_estate(mols_list):
    vsa_fp = VSAFingerprint(variant="all", n_jobs=-1)
    X_skfp = vsa_fp.transform(mols_list)

    X_slogp = np.array([SlogP_VSA_(mol) for mol in mols_list])
    X_smr = np.array([SMR_VSA_(mol) for mol in mols_list])
    X_peoe = np.array([PEOE_VSA_(mol) for mol in mols_list])
    X_estate = np.array([EState_VSA_(mol) for mol in mols_list])

    X_rdkit = np.column_stack((X_slogp, X_smr, X_peoe, X_estate))

    assert np.allclose(X_skfp, X_rdkit)
    assert X_skfp.shape == (len(mols_list), 47)
    assert np.issubdtype(X_skfp.dtype, np.floating)


def test_vsa_fingerprint_variants(mols_list):
    for variant, expected_num_cols in [
        ("SlogP", 12),
        ("SMR", 10),
        ("PEOE", 14),
        ("EState", 11),
        ("all_original", 36),
        ("all", 47),
    ]:
        vsa_fp = VSAFingerprint(variant=variant, n_jobs=-1)
        X_skfp = vsa_fp.transform(mols_list)

        assert X_skfp.shape == (len(mols_list), expected_num_cols)
        assert np.issubdtype(X_skfp.dtype, np.floating)


def test_vsa_sparse_fingerprint(mols_list):
    vsa_fp = VSAFingerprint(sparse=True, n_jobs=-1)
    X_skfp = vsa_fp.transform(mols_list)

    X_slogp = np.array([SlogP_VSA_(mol) for mol in mols_list])
    X_smr = np.array([SMR_VSA_(mol) for mol in mols_list])
    X_peoe = np.array([PEOE_VSA_(mol) for mol in mols_list])

    X_rdkit = csr_array(np.column_stack((X_slogp, X_smr, X_peoe)))

    assert np.allclose(X_skfp.data, X_rdkit.data)
    assert X_skfp.shape == (len(mols_list), 36)
    assert np.issubdtype(X_skfp.dtype, np.floating)
