import numpy as np
import pytest
from rdkit.Chem.rdMHFPFingerprint import MHFPEncoder
from scipy.sparse import csr_array
from sklearn.utils._param_validation import InvalidParameterError

from skfp.fingerprints import SECFPFingerprint


def test_secfp_fingerprint(smiles_list, mols_list):
    secfp_fp = SECFPFingerprint(sparse=False, n_jobs=-1)
    X_skfp = secfp_fp.transform(smiles_list)

    encoder = MHFPEncoder(2048, 0)
    X_rdkit = np.array([encoder.EncodeSECFPMol(x) for x in mols_list])

    assert np.array_equal(X_skfp, X_rdkit)
    assert X_skfp.shape == (len(smiles_list), secfp_fp.fp_size)


def test_secfp_sparse_fingerprint(smiles_list, mols_list):
    secfp_fp = SECFPFingerprint(sparse=True, n_jobs=-1)
    X_skfp = secfp_fp.transform(smiles_list)

    encoder = MHFPEncoder(2048, 0)
    X_rdkit = csr_array([encoder.EncodeSECFPMol(x) for x in mols_list])

    assert np.array_equal(X_skfp.data, X_rdkit.data)
    assert X_skfp.shape == (len(smiles_list), secfp_fp.fp_size)


def test_secfp_wrong_radii(smiles_list):
    secfp_fp = SECFPFingerprint(min_radius=3, radius=2)
    with pytest.raises(InvalidParameterError):
        secfp_fp.transform(smiles_list)
