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


def test_electroshape_ignore_errors():
    organometallics = [
        "CCCC[Li]",
        "CC[Zn]CC",
        "C[Al](C)C",
        "CCCC[SnH](CCCC)CCCC",
    ]
    non_metallics = [
        # benzene
        "c1ccccc1",
        # ibuprofen
        "CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O",
        # caffeine
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        # nicotine
        "c1ncccc1[C@@H]2CCCN2C",
        # Venlafaxine
        "OC2(C(c1ccc(OC)cc1)CN(C)C)CCCCC2",
        # Chlordiazepoxide
        "ClC1=CC2=C(N=C(NC)C[N+]([O-])=C2C3=CC=CC=C3)C=C1",
    ]
    all_smiles = organometallics + non_metallics
    all_mols = MolFromSmilesTransformer().transform(all_smiles)
    all_mols = ConformerGenerator().transform(all_mols)

    electroshape_fp = ElectroShapeFingerprint(errors="ignore", n_jobs=-1)
    X_skfp = electroshape_fp.transform(all_mols)

    assert len(X_skfp) == len(all_mols)
