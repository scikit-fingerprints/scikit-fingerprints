import numpy as np
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D
from rdkit.Chem.Pharm2D.Generate import Gen2DFingerprint
from scipy.sparse import csr_array

from skfp.fingerprints import PharmacophoreFingerprint


def test_pharmacophore_bit_fingerprint(smallest_smiles_list, smallest_mols_list):
    pharmacophore_fp = PharmacophoreFingerprint(sparse=False, n_jobs=-1)
    X_skfp = pharmacophore_fp.transform(smallest_smiles_list)

    factory = Gobbi_Pharm2D.factory
    X_rdkit = np.array([Gen2DFingerprint(mol, factory) for mol in smallest_mols_list])

    assert np.array_equal(X_skfp, X_rdkit)


def test_pharmacophore_sparse_bit_fingerprint(smallest_smiles_list, smallest_mols_list):
    pharmacophore_fp = PharmacophoreFingerprint(sparse=True, n_jobs=-1)
    X_skfp = pharmacophore_fp.transform(smallest_smiles_list)

    factory = Gobbi_Pharm2D.factory
    X_rdkit = csr_array([Gen2DFingerprint(mol, factory) for mol in smallest_mols_list])

    assert np.array_equal(X_skfp.data, X_rdkit.data)
