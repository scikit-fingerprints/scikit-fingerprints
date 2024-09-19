import numpy as np
import pytest

from skfp.fingerprints import ElectroShapeFingerprint
from skfp.preprocessing import ConformerGenerator, MolFromSmilesTransformer


@pytest.fixture
def mols_conformers_3_plus_atoms(mols_conformers_list):
    # electroshape descriptor requires at least 3 atoms to work
    return [mol for mol in mols_conformers_list if mol.GetNumAtoms() >= 3]


def test_electroshape_bit_fingerprint(mols_conformers_3_plus_atoms):
    electroshape_fp = ElectroShapeFingerprint(n_jobs=-1)
    X_skfp = electroshape_fp.transform(mols_conformers_3_plus_atoms)

    assert X_skfp.shape == (len(mols_conformers_3_plus_atoms), 15)
    assert np.issubdtype(X_skfp.dtype, np.floating)
    assert not np.any(np.isnan(X_skfp))


def test_electroshape_bit_fingerprint_transform_x_y(mols_conformers_3_plus_atoms):
    y = np.arange(len(mols_conformers_3_plus_atoms))

    electroshape_fp = ElectroShapeFingerprint(n_jobs=-1)
    X_skfp, y_skfp = electroshape_fp.transform_x_y(mols_conformers_3_plus_atoms, y)

    assert X_skfp.shape == (len(mols_conformers_3_plus_atoms), 15)
    assert np.issubdtype(X_skfp.dtype, np.floating)
    assert not np.any(np.isnan(X_skfp))
    assert len(X_skfp) == len(y_skfp)


def test_electroshape_charge_models(mols_conformers_3_plus_atoms):
    for charge_model in ["Gasteiger", "MMFF94", "formal"]:
        fp = ElectroShapeFingerprint(
            partial_charge_model=charge_model, charge_errors="zero"
        )
        fp.transform(mols_conformers_3_plus_atoms)


def test_mmff94_error():
    mols = ["[H]O[As]=O"]
    mols = MolFromSmilesTransformer().transform(mols)
    mols = ConformerGenerator().transform(mols)
    with pytest.raises(ValueError) as exc_info:
        ElectroShapeFingerprint(partial_charge_model="MMFF94").transform(mols)

    assert "Failed to compute at least one atom partial charge" in str(exc_info)

    # this, on the other hand, should not error
    ElectroShapeFingerprint(charge_errors="zero").transform(mols)


def test_electroshape_ignore_errors(mols_conformers_list, mols_conformers_3_plus_atoms):
    usr_fp = ElectroShapeFingerprint(errors="ignore", n_jobs=-1)
    X_skfp = usr_fp.transform(mols_conformers_list)
    X_skfp_3_plus_atoms = usr_fp.transform(mols_conformers_3_plus_atoms)

    assert np.allclose(X_skfp, X_skfp_3_plus_atoms)
