import numpy as np
import pytest
from numpy.testing import assert_equal
from rdkit.Chem import Get3DDistanceMatrix, Mol
from rdkit.Chem.ChemicalFeatures import BuildFeatureFactoryFromString
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D
from rdkit.Chem.Pharm2D.Generate import Gen2DFingerprint
from rdkit.Chem.Pharm2D.SigFactory import SigFactory
from scipy.sparse import csr_array

from skfp.fingerprints import PharmacophoreFingerprint


def test_pharmacophore_raw_bits_fingerprint(smallest_smiles_list, smallest_mols_list):
    pharmacophore_fp = PharmacophoreFingerprint(n_jobs=-1)
    X_skfp = pharmacophore_fp.transform(smallest_smiles_list)

    X_rdkit = np.array(_get_rdkit_pharmacophore_fp(smallest_mols_list))

    assert_equal(X_skfp, X_rdkit)
    assert_equal(X_skfp.shape, (len(smallest_smiles_list), 39972))
    assert X_skfp.dtype == np.uint8
    assert np.all(np.isin(X_skfp, [0, 1]))


def test_pharmacophore_raw_bits_3D_fingerprint(mols_conformers_list):
    pharmacophore_fp = PharmacophoreFingerprint(use_3D=True, n_jobs=-1)
    X_skfp = pharmacophore_fp.transform(mols_conformers_list)

    X_rdkit = np.array(_get_rdkit_pharmacophore_fp(mols_conformers_list, use_3D=True))

    assert_equal(X_skfp, X_rdkit)
    assert_equal(X_skfp.shape, (len(mols_conformers_list), 39972))
    assert X_skfp.dtype == np.uint8
    assert np.all(np.isin(X_skfp, [0, 1]))


def test_pharmacophore_bit_fingerprint(smallest_smiles_list, smallest_mols_list):
    pharmacophore_fp = PharmacophoreFingerprint(variant="folded", n_jobs=-1)
    X_skfp = pharmacophore_fp.transform(smallest_smiles_list)

    X_rdkit = pharmacophore_fp._hash_fingerprint_bits(
        _get_rdkit_pharmacophore_fp(smallest_mols_list),
        fp_size=pharmacophore_fp.fp_size,
        count=False,
        sparse=False,
    )

    assert_equal(X_skfp, X_rdkit)
    assert_equal(X_skfp.shape, (len(smallest_smiles_list), pharmacophore_fp.fp_size))
    assert X_skfp.dtype == np.uint8
    assert np.all(np.isin(X_skfp, [0, 1]))


def test_pharmacophore_bit_3D_fingerprint(mols_conformers_list):
    pharmacophore_fp = PharmacophoreFingerprint(
        variant="folded", use_3D=True, n_jobs=-1
    )
    X_skfp = pharmacophore_fp.transform(mols_conformers_list)

    X_rdkit = pharmacophore_fp._hash_fingerprint_bits(
        _get_rdkit_pharmacophore_fp(mols_conformers_list, use_3D=True),
        fp_size=pharmacophore_fp.fp_size,
        count=False,
        sparse=False,
    )

    assert_equal(X_skfp, X_rdkit)
    assert_equal(X_skfp.shape, (len(mols_conformers_list), pharmacophore_fp.fp_size))
    assert X_skfp.dtype == np.uint8
    assert np.all(np.isin(X_skfp, [0, 1]))


def test_pharmacophore_count_fingerprint(smallest_smiles_list, smallest_mols_list):
    pharmacophore_fp = PharmacophoreFingerprint(variant="folded", count=True, n_jobs=-1)
    X_skfp = pharmacophore_fp.transform(smallest_smiles_list)

    X_rdkit = pharmacophore_fp._hash_fingerprint_bits(
        _get_rdkit_pharmacophore_fp(smallest_mols_list, count=True),
        fp_size=pharmacophore_fp.fp_size,
        count=True,
        sparse=False,
    )

    assert_equal(X_skfp, X_rdkit)
    assert_equal(X_skfp.shape, (len(smallest_smiles_list), pharmacophore_fp.fp_size))
    assert X_skfp.dtype == np.uint32
    assert np.all(X_skfp >= 0)


def test_pharmacophore_count_3D_fingerprint(mols_conformers_list):
    pharmacophore_fp = PharmacophoreFingerprint(
        variant="folded", use_3D=True, count=True, n_jobs=-1
    )
    X_skfp = pharmacophore_fp.transform(mols_conformers_list)

    X_rdkit = pharmacophore_fp._hash_fingerprint_bits(
        _get_rdkit_pharmacophore_fp(mols_conformers_list, count=True, use_3D=True),
        fp_size=pharmacophore_fp.fp_size,
        count=True,
        sparse=False,
    )

    assert_equal(X_skfp, X_rdkit)
    assert_equal(X_skfp.shape, (len(mols_conformers_list), pharmacophore_fp.fp_size))
    assert X_skfp.dtype == np.uint32
    assert np.all(X_skfp >= 0)


def test_pharmacophore_raw_bits_sparse_fingerprint(
    smallest_smiles_list, smallest_mols_list
):
    pharmacophore_fp = PharmacophoreFingerprint(sparse=True, n_jobs=-1)
    X_skfp = pharmacophore_fp.transform(smallest_smiles_list)

    X_rdkit = csr_array(_get_rdkit_pharmacophore_fp(smallest_mols_list))

    assert_equal(X_skfp.data, X_rdkit.data)
    assert_equal(X_skfp.shape, (len(smallest_smiles_list), 39972))
    assert X_skfp.dtype == np.uint8
    assert np.all(X_skfp.data == 1)


def test_pharmacophore_raw_bits_3D_sparse_fingerprint(mols_conformers_list):
    pharmacophore_fp = PharmacophoreFingerprint(use_3D=True, sparse=True, n_jobs=-1)
    X_skfp = pharmacophore_fp.transform(mols_conformers_list)

    X_rdkit = csr_array(_get_rdkit_pharmacophore_fp(mols_conformers_list, use_3D=True))

    assert_equal(X_skfp.data, X_rdkit.data)
    assert_equal(X_skfp.shape, (len(mols_conformers_list), 39972))
    assert X_skfp.dtype == np.uint8
    assert np.all(X_skfp.data == 1)


def test_pharmacophore_bit_sparse_fingerprint(smallest_smiles_list, smallest_mols_list):
    pharmacophore_fp = PharmacophoreFingerprint(
        variant="folded", sparse=True, n_jobs=-1
    )
    X_skfp = pharmacophore_fp.transform(smallest_smiles_list)

    X_rdkit = pharmacophore_fp._hash_fingerprint_bits(
        _get_rdkit_pharmacophore_fp(smallest_mols_list),
        fp_size=pharmacophore_fp.fp_size,
        count=False,
        sparse=True,
    )

    assert_equal(X_skfp.data, X_rdkit.data)
    assert_equal(X_skfp.shape, (len(smallest_smiles_list), pharmacophore_fp.fp_size))
    assert X_skfp.dtype == np.uint8
    assert np.all(X_skfp.data == 1)


def test_pharmacophore_bit_3D_sparse_fingerprint(mols_conformers_list):
    pharmacophore_fp = PharmacophoreFingerprint(
        variant="folded", use_3D=True, sparse=True, n_jobs=-1
    )
    X_skfp = pharmacophore_fp.transform(mols_conformers_list)

    X_rdkit = pharmacophore_fp._hash_fingerprint_bits(
        _get_rdkit_pharmacophore_fp(mols_conformers_list, use_3D=True),
        fp_size=pharmacophore_fp.fp_size,
        count=False,
        sparse=True,
    )

    assert_equal(X_skfp.data, X_rdkit.data)
    assert_equal(X_skfp.shape, (len(mols_conformers_list), pharmacophore_fp.fp_size))
    assert X_skfp.dtype == np.uint8
    assert np.all(X_skfp.data == 1)


def test_pharmacophore_count_sparse_fingerprint(
    smallest_smiles_list, smallest_mols_list
):
    pharmacophore_fp = PharmacophoreFingerprint(
        variant="folded", count=True, sparse=True, n_jobs=-1
    )
    X_skfp = pharmacophore_fp.transform(smallest_smiles_list)

    X_rdkit = pharmacophore_fp._hash_fingerprint_bits(
        _get_rdkit_pharmacophore_fp(smallest_mols_list, count=True),
        fp_size=pharmacophore_fp.fp_size,
        count=True,
        sparse=True,
    )

    assert_equal(X_skfp.data, X_rdkit.data)
    assert_equal(X_skfp.shape, (len(smallest_smiles_list), pharmacophore_fp.fp_size))
    assert X_skfp.dtype == np.uint32
    assert np.all(X_skfp.data > 0)


def test_pharmacophore_count_3D_sparse_fingerprint(mols_conformers_list):
    pharmacophore_fp = PharmacophoreFingerprint(
        variant="folded", use_3D=True, count=True, sparse=True, n_jobs=-1
    )
    X_skfp = pharmacophore_fp.transform(mols_conformers_list)

    X_rdkit = pharmacophore_fp._hash_fingerprint_bits(
        _get_rdkit_pharmacophore_fp(mols_conformers_list, count=True, use_3D=True),
        fp_size=pharmacophore_fp.fp_size,
        count=True,
        sparse=True,
    )

    assert_equal(X_skfp.data, X_rdkit.data)
    assert_equal(X_skfp.shape, (len(mols_conformers_list), pharmacophore_fp.fp_size))
    assert X_skfp.dtype == np.uint32
    assert np.all(X_skfp.data > 0)


def test_pharmacophore_2_2_points(smallest_smiles_list, smallest_mols_list):
    pharmacophore_fp = PharmacophoreFingerprint(min_points=2, max_points=2)
    X_skfp = pharmacophore_fp.transform(smallest_smiles_list)

    X_rdkit = np.array(
        _get_rdkit_pharmacophore_fp(smallest_mols_list, min_points=2, max_points=2)
    )

    assert_equal(X_skfp, X_rdkit)
    assert_equal(X_skfp.shape, (len(smallest_smiles_list), 252))


def test_pharmacophore_3_3_points(smallest_smiles_list, smallest_mols_list):
    pharmacophore_fp = PharmacophoreFingerprint(min_points=3, max_points=3)
    X_skfp = pharmacophore_fp.transform(smallest_smiles_list)

    X_rdkit = np.array(
        _get_rdkit_pharmacophore_fp(smallest_mols_list, min_points=3, max_points=3)
    )

    assert_equal(X_skfp, X_rdkit)
    assert_equal(X_skfp.shape, (len(smallest_smiles_list), 39720))


def test_pharmacophore_wrong_n_points():
    with pytest.raises(ValueError) as exc_info:
        PharmacophoreFingerprint(min_points=3, max_points=2)
    assert "min_points <= max_points" in str(exc_info)

    with pytest.raises(ValueError) as exc_info:
        PharmacophoreFingerprint(max_points=4)
    assert "min_points and max_points must be 2 or 3" in str(exc_info)


def _get_rdkit_pharmacophore_fp(
    mols: list[Mol],
    min_points: int = 2,
    max_points: int = 3,
    count: bool = False,
    use_3D: bool = False,
) -> list:
    atom_features = BuildFeatureFactoryFromString(Gobbi_Pharm2D.fdef)
    factory = SigFactory(
        atom_features,
        minPointCount=min_points,
        maxPointCount=max_points,
        useCounts=count,
    )
    factory.SetBins(Gobbi_Pharm2D.defaultBins)
    factory.Init()

    dists_3d_list = [
        Get3DDistanceMatrix(mol, confId=mol.GetIntProp("conf_id")) if use_3D else None
        for mol in mols
    ]
    return [
        Gen2DFingerprint(mol, factory, dMat=dists_3d)
        for mol, dists_3d in zip(mols, dists_3d_list, strict=False)
    ]
