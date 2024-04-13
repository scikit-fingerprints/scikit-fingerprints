import numpy as np
from rdkit.Chem import Get3DDistanceMatrix
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D
from rdkit.Chem.Pharm2D.Generate import Gen2DFingerprint
from scipy.sparse import csr_array

from skfp.fingerprints import PharmacophoreFingerprint


def test_pharmacophore_raw_bits_fingerprint(smallest_smiles_list, smallest_mols_list):
    pharmacophore_fp = PharmacophoreFingerprint(sparse=False, n_jobs=1)
    X_skfp = pharmacophore_fp.transform(smallest_smiles_list)

    factory = Gobbi_Pharm2D.factory
    X_rdkit = np.array([Gen2DFingerprint(x, factory) for x in smallest_mols_list])

    assert np.array_equal(X_skfp, X_rdkit)
    assert X_skfp.shape == (len(smallest_smiles_list), 39972)
    assert X_skfp.dtype == np.uint8
    assert np.all(np.isin(X_skfp, [0, 1]))


def test_pharmacophore_raw_bits_3D_fingerprint(mols_conformers_list):
    pharmacophore_fp = PharmacophoreFingerprint(use_3D=True, sparse=False, n_jobs=1)
    X_skfp = pharmacophore_fp.transform(mols_conformers_list)

    factory = Gobbi_Pharm2D.factory
    X_rdkit = np.array(
        [
            Gen2DFingerprint(
                mol,
                factory,
                dMat=Get3DDistanceMatrix(mol, confId=mol.GetIntProp("conf_id")),
            )
            for mol in mols_conformers_list
        ]
    )

    assert np.array_equal(X_skfp, X_rdkit)
    assert X_skfp.shape == (len(mols_conformers_list), 39972)
    assert X_skfp.dtype == np.uint8
    assert np.all(np.isin(X_skfp, [0, 1]))


def test_pharmacophore_bit_fingerprint(smallest_smiles_list, smallest_mols_list):
    pharmacophore_fp = PharmacophoreFingerprint(variant="bit", sparse=False, n_jobs=1)
    X_skfp = pharmacophore_fp.transform(smallest_smiles_list)

    factory = Gobbi_Pharm2D.factory
    X_rdkit = [Gen2DFingerprint(x, factory) for x in smallest_mols_list]
    X_rdkit = pharmacophore_fp._hash_fingerprint_bits(
        X_rdkit, fp_size=pharmacophore_fp.fp_size, count=False, sparse=False
    )

    assert np.array_equal(X_skfp, X_rdkit)
    assert X_skfp.shape == (len(smallest_smiles_list), pharmacophore_fp.fp_size)
    assert X_skfp.dtype == np.uint8
    assert np.all(np.isin(X_skfp, [0, 1]))


def test_pharmacophore_bit_3D_fingerprint(mols_conformers_list):
    pharmacophore_fp = PharmacophoreFingerprint(
        variant="bit", use_3D=True, sparse=False, n_jobs=1
    )
    X_skfp = pharmacophore_fp.transform(mols_conformers_list)

    factory = Gobbi_Pharm2D.factory
    X_rdkit = [
        Gen2DFingerprint(
            mol,
            factory,
            dMat=Get3DDistanceMatrix(mol, confId=mol.GetIntProp("conf_id")),
        )
        for mol in mols_conformers_list
    ]
    X_rdkit = pharmacophore_fp._hash_fingerprint_bits(
        X_rdkit, fp_size=pharmacophore_fp.fp_size, count=False, sparse=False
    )

    assert np.array_equal(X_skfp, X_rdkit)
    assert X_skfp.shape == (len(mols_conformers_list), pharmacophore_fp.fp_size)
    assert X_skfp.dtype == np.uint8
    assert np.all(np.isin(X_skfp, [0, 1]))


def test_pharmacophore_count_fingerprint(smallest_smiles_list, smallest_mols_list):
    pharmacophore_fp = PharmacophoreFingerprint(variant="count", sparse=False, n_jobs=1)
    X_skfp = pharmacophore_fp.transform(smallest_smiles_list)

    factory = Gobbi_Pharm2D.factory
    X_rdkit = [Gen2DFingerprint(x, factory) for x in smallest_mols_list]
    X_rdkit = pharmacophore_fp._hash_fingerprint_bits(
        X_rdkit, fp_size=pharmacophore_fp.fp_size, count=True, sparse=False
    )

    assert np.array_equal(X_skfp, X_rdkit)
    assert X_skfp.shape == (len(smallest_smiles_list), pharmacophore_fp.fp_size)
    assert X_skfp.dtype == np.uint32
    assert np.all(X_skfp >= 0)


def test_pharmacophore_count_3D_fingerprint(mols_conformers_list):
    pharmacophore_fp = PharmacophoreFingerprint(
        variant="count", use_3D=True, sparse=False, n_jobs=1
    )
    X_skfp = pharmacophore_fp.transform(mols_conformers_list)

    factory = Gobbi_Pharm2D.factory
    X_rdkit = [
        Gen2DFingerprint(
            mol,
            factory,
            dMat=Get3DDistanceMatrix(mol, confId=mol.GetIntProp("conf_id")),
        )
        for mol in mols_conformers_list
    ]
    X_rdkit = pharmacophore_fp._hash_fingerprint_bits(
        X_rdkit, fp_size=pharmacophore_fp.fp_size, count=True, sparse=False
    )

    assert np.array_equal(X_skfp, X_rdkit)
    assert X_skfp.shape == (len(mols_conformers_list), pharmacophore_fp.fp_size)
    assert X_skfp.dtype == np.uint32
    assert np.all(X_skfp >= 0)


def test_pharmacophore_raw_bits_sparse_fingerprint(
    smallest_smiles_list, smallest_mols_list
):
    pharmacophore_fp = PharmacophoreFingerprint(sparse=True, n_jobs=-1)
    X_skfp = pharmacophore_fp.transform(smallest_smiles_list)

    factory = Gobbi_Pharm2D.factory
    X_rdkit = csr_array([Gen2DFingerprint(mol, factory) for mol in smallest_mols_list])

    assert np.array_equal(X_skfp.data, X_rdkit.data)
    assert X_skfp.shape == (len(smallest_smiles_list), 39972)
    assert X_skfp.dtype == np.uint8
    assert np.all(X_skfp.data == 1)


def test_pharmacophore_raw_bits_3D_sparse_fingerprint(mols_conformers_list):
    pharmacophore_fp = PharmacophoreFingerprint(use_3D=True, sparse=True, n_jobs=1)
    X_skfp = pharmacophore_fp.transform(mols_conformers_list)

    factory = Gobbi_Pharm2D.factory
    X_rdkit = csr_array(
        [
            Gen2DFingerprint(
                mol,
                factory,
                dMat=Get3DDistanceMatrix(mol, confId=mol.GetIntProp("conf_id")),
            )
            for mol in mols_conformers_list
        ]
    )

    assert np.array_equal(X_skfp.data, X_rdkit.data)
    assert X_skfp.shape == (len(mols_conformers_list), 39972)
    assert X_skfp.dtype == np.uint8
    assert np.all(X_skfp.data == 1)


def test_pharmacophore_bit_sparse_fingerprint(smallest_smiles_list, smallest_mols_list):
    pharmacophore_fp = PharmacophoreFingerprint(variant="bit", sparse=True, n_jobs=1)
    X_skfp = pharmacophore_fp.transform(smallest_smiles_list)

    factory = Gobbi_Pharm2D.factory
    X_rdkit = [Gen2DFingerprint(x, factory) for x in smallest_mols_list]
    X_rdkit = pharmacophore_fp._hash_fingerprint_bits(
        X_rdkit, fp_size=pharmacophore_fp.fp_size, count=False, sparse=True
    )

    assert np.array_equal(X_skfp.data, X_rdkit.data)
    assert X_skfp.shape == (len(smallest_smiles_list), pharmacophore_fp.fp_size)
    assert X_skfp.dtype == np.uint8
    assert np.all(X_skfp.data == 1)


def test_pharmacophore_bit_3D_sparse_fingerprint(mols_conformers_list):
    pharmacophore_fp = PharmacophoreFingerprint(
        variant="bit", use_3D=True, sparse=True, n_jobs=1
    )
    X_skfp = pharmacophore_fp.transform(mols_conformers_list)

    factory = Gobbi_Pharm2D.factory
    X_rdkit = [
        Gen2DFingerprint(
            mol,
            factory,
            dMat=Get3DDistanceMatrix(mol, confId=mol.GetIntProp("conf_id")),
        )
        for mol in mols_conformers_list
    ]
    X_rdkit = pharmacophore_fp._hash_fingerprint_bits(
        X_rdkit, fp_size=pharmacophore_fp.fp_size, count=False, sparse=True
    )

    assert np.array_equal(X_skfp.data, X_rdkit.data)
    assert X_skfp.shape == (len(mols_conformers_list), pharmacophore_fp.fp_size)
    assert X_skfp.dtype == np.uint8
    assert np.all(X_skfp.data == 1)


def test_pharmacophore_count_sparse_fingerprint(
    smallest_smiles_list, smallest_mols_list
):
    pharmacophore_fp = PharmacophoreFingerprint(variant="count", sparse=True, n_jobs=1)
    X_skfp = pharmacophore_fp.transform(smallest_smiles_list)

    factory = Gobbi_Pharm2D.factory
    X_rdkit = [Gen2DFingerprint(x, factory) for x in smallest_mols_list]
    X_rdkit = pharmacophore_fp._hash_fingerprint_bits(
        X_rdkit, fp_size=pharmacophore_fp.fp_size, count=True, sparse=True
    )

    assert np.array_equal(X_skfp.data, X_rdkit.data)
    assert X_skfp.shape == (len(smallest_smiles_list), pharmacophore_fp.fp_size)
    assert X_skfp.dtype == np.uint32
    assert np.all(X_skfp.data > 0)


def test_pharmacophore_count_3D_sparse_fingerprint(mols_conformers_list):
    pharmacophore_fp = PharmacophoreFingerprint(
        variant="count", use_3D=True, sparse=True, n_jobs=1
    )
    X_skfp = pharmacophore_fp.transform(mols_conformers_list)

    factory = Gobbi_Pharm2D.factory
    X_rdkit = [
        Gen2DFingerprint(
            mol,
            factory,
            dMat=Get3DDistanceMatrix(mol, confId=mol.GetIntProp("conf_id")),
        )
        for mol in mols_conformers_list
    ]
    X_rdkit = pharmacophore_fp._hash_fingerprint_bits(
        X_rdkit, fp_size=pharmacophore_fp.fp_size, count=True, sparse=True
    )

    assert np.array_equal(X_skfp.data, X_rdkit.data)
    assert X_skfp.shape == (len(mols_conformers_list), pharmacophore_fp.fp_size)
    assert X_skfp.dtype == np.uint32
    assert np.all(X_skfp.data > 0)
