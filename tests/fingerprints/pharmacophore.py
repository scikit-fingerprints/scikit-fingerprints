import numpy as np
from rdkit.Chem import AddHs, Get3DDistanceMatrix
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D
from rdkit.Chem.Pharm2D.Generate import Gen2DFingerprint
from rdkit.Chem.rdDistGeom import EmbedMolecule
from scipy.sparse import csr_array

from skfp.fingerprints import PharmacophoreFingerprint


def test_pharmacophore_bit_fingerprint(smiles_list, mols_list):
    pharmacophore_fp = PharmacophoreFingerprint(sparse=False, n_jobs=-1)
    X_skfp = pharmacophore_fp.transform(smiles_list)

    factory = Gobbi_Pharm2D.factory
    X_rdkit = np.array([Gen2DFingerprint(mol, factory) for mol in mols_list])

    assert np.array_equal(X_skfp, X_rdkit)


def test_3D_pharmacophore_bit_fingerprint(smiles_list, mols_list):
    pharmacophore_fp = PharmacophoreFingerprint(
        use_3D=True, sparse=False, n_jobs=-1
    )
    X_skfp = pharmacophore_fp.transform(smiles_list)

    mols_list = [AddHs(mol) for mol in mols_list]
    factory = Gobbi_Pharm2D.factory
    X_rdkit = np.array([Gen2DFingerprint(mol, factory) for mol in mols_list])

    assert np.array_equal(X_skfp, X_rdkit)


def test_pharmacophore_sparse_bit_fingerprint(smiles_list, mols_list):
    pharmacophore_fp = PharmacophoreFingerprint(sparse=True, n_jobs=-1)
    X_skfp = pharmacophore_fp.transform(smiles_list)

    factory = Gobbi_Pharm2D.factory
    X_rdkit = csr_array([Gen2DFingerprint(mol, factory) for mol in mols_list])

    assert np.array_equal(X_skfp.data, X_rdkit.data)


def test_3D_pharmacophore_sparse_bit_fingerprint(smiles_list, mols_list):
    pharmacophore_fp = PharmacophoreFingerprint(
        use_3D=True, sparse=True, n_jobs=-1
    )
    X_skfp = pharmacophore_fp.transform(smiles_list)

    mols_list = [AddHs(mol) for mol in mols_list]
    for mol in mols_list:
        EmbedMolecule(mol)
    factory = Gobbi_Pharm2D.factory
    X_rdkit = csr_array(
        [
            Gen2DFingerprint(mol, factory, dMat=Get3DDistanceMatrix(mol))
            for mol in mols_list
        ]
    )

    assert np.array_equal(X_skfp.data, X_rdkit.data)
