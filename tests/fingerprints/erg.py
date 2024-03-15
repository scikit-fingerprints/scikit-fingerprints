import numpy as np
from rdkit.Chem.rdReducedGraphs import GetErGFingerprint
from scipy.sparse import csr_array

from helpers import sparse_equal
from skfp import ERGFingerprint


def test_erg_bit_fingerprint(smiles_list, mols_list):
    erg_fp = ERGFingerprint(sparse=False, n_jobs=-1)
    X_skfp = erg_fp.transform(smiles_list)

    X_rdkit = np.array([GetErGFingerprint(mol) for mol in mols_list])

    assert np.array_equal(X_skfp, X_rdkit)


def test_erg_sparse_bit_fingerprint(smiles_list, mols_list):
    erg_fp = ERGFingerprint(sparse=True, n_jobs=-1)
    X_skfp = erg_fp.transform(smiles_list)

    X_rdkit = csr_array([GetErGFingerprint(mol) for mol in mols_list])

    assert sparse_equal(X_skfp, X_rdkit)
