import numpy as np
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdReducedGraphs import GetErGFingerprint
from scipy.sparse import csr_array

from skfp.fingerprints import ERGFingerprint

"""
Note: for some unknown reason passing mols_list from the global fixture sometimes
does not work for ErG fingerprints. This happens only for those tests, and not for
any other fingerprint. Creating molecules from SMILES separately here works.
"""


def test_erg_fingerprint(smiles_list):
    erg_fp = ERGFingerprint(sparse=False, n_jobs=-1)
    X_skfp = erg_fp.transform(smiles_list)

    mols_list = [MolFromSmiles(smi) for smi in smiles_list]
    X_rdkit = np.array([GetErGFingerprint(mol) for mol in mols_list])

    assert np.array_equal(X_skfp, X_rdkit)


def test_erg_sparse_fingerprint(smiles_list):
    erg_fp = ERGFingerprint(sparse=True, n_jobs=-1)
    X_skfp = erg_fp.transform(smiles_list)

    mols_list = [MolFromSmiles(smi) for smi in smiles_list]
    X_rdkit = csr_array([GetErGFingerprint(mol) for mol in mols_list])

    assert np.all(np.isclose(X_skfp.data, X_rdkit.data))
