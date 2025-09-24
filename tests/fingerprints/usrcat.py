import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from rdkit.Chem.rdMolDescriptors import GetUSRCAT

from skfp.fingerprints import USRCATFingerprint


@pytest.fixture
def mols_conformers_3_plus_atoms(mols_conformers_list):
    # USRCAT descriptor requires at least 3 atoms to work
    return [mol for mol in mols_conformers_list if mol.GetNumAtoms() >= 3]


def test_usrcat_bit_fingerprint(mols_conformers_3_plus_atoms):
    usrcat_fp = USRCATFingerprint()
    X_skfp = usrcat_fp.transform(mols_conformers_3_plus_atoms)

    X_rdkit = np.array(
        [
            GetUSRCAT(mol, confId=mol.GetIntProp("conf_id"))
            for mol in mols_conformers_3_plus_atoms
        ]
    )

    assert_allclose(X_skfp, X_rdkit, atol=1e-3)
    assert_equal(X_skfp.shape, (len(mols_conformers_3_plus_atoms), 60))
    assert np.issubdtype(X_skfp.dtype, np.floating)


def test_usrcat_bit_fingerprint_transform_x_y(mols_conformers_3_plus_atoms):
    labels = np.arange(len(mols_conformers_3_plus_atoms))

    usrcat_fp = USRCATFingerprint()
    X_skfp, y_skfp = usrcat_fp.transform_x_y(mols_conformers_3_plus_atoms, labels)

    X_rdkit = []
    y_rdkit = []
    for mol, y in zip(mols_conformers_3_plus_atoms, labels, strict=False):
        mol_fp = GetUSRCAT(mol, confId=mol.GetIntProp("conf_id"))
        X_rdkit.append(mol_fp)
        y_rdkit.append(y)

    X_rdkit = np.array(X_rdkit)
    y_rdkit = np.array(y_rdkit)

    assert_allclose(X_skfp, X_rdkit, atol=1e-3)
    assert_equal(X_skfp.shape, (len(mols_conformers_3_plus_atoms), 60))
    assert np.issubdtype(X_skfp.dtype, np.floating)
    assert_equal(y_skfp, y_rdkit)


def test_usrcat_ignore_errors(mols_conformers_list, mols_conformers_3_plus_atoms):
    usrcat_fp = USRCATFingerprint(errors="ignore", n_jobs=-1)
    X_skfp = usrcat_fp.transform(mols_conformers_list)
    X_skfp_3_plus_atoms = usrcat_fp.transform(mols_conformers_3_plus_atoms)

    assert_allclose(X_skfp, X_skfp_3_plus_atoms)


def test_usrcat_copy(mols_conformers_3_plus_atoms):
    # smoke test, should not throw an error
    labels = np.ones(len(mols_conformers_3_plus_atoms))

    usrcat_fp = USRCATFingerprint(errors="ignore", n_jobs=-1)
    fps, labels_out = usrcat_fp.transform_x_y(
        mols_conformers_3_plus_atoms, labels, copy=True
    )

    np.testing.assert_array_equal(labels, labels_out)
    assert labels is not labels_out


def test_usrcat_feature_names():
    usrcat_fp = USRCATFingerprint()
    feature_names = usrcat_fp.get_feature_names_out()

    assert_equal(len(feature_names), usrcat_fp.n_features_out)
    assert_equal(len(feature_names), len(set(feature_names)))

    assert_equal(feature_names[0], "all_centroid_dists_mean")
    assert_equal(feature_names[12], "hydrophobic_centroid_dists_mean")
    assert_equal(feature_names[24], "aromatic_centroid_dists_mean")
    assert_equal(feature_names[36], "hydrogen_bond_donor_centroid_dists_mean")
    assert_equal(feature_names[48], "hydrogen_bond_acceptor_centroid_dists_mean")

    assert_equal(
        feature_names[11], "all_farthest_atom_from_farthest_to_centroid_cubic_root"
    )
    assert_equal(
        feature_names[23],
        "hydrophobic_farthest_atom_from_farthest_to_centroid_cubic_root",
    )
    assert_equal(
        feature_names[35], "aromatic_farthest_atom_from_farthest_to_centroid_cubic_root"
    )
    assert_equal(
        feature_names[47],
        "hydrogen_bond_donor_farthest_atom_from_farthest_to_centroid_cubic_root",
    )
    assert_equal(
        feature_names[59],
        "hydrogen_bond_acceptor_farthest_atom_from_farthest_to_centroid_cubic_root",
    )
