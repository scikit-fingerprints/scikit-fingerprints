import pytest

from skfp.fingerprints import AtomPairFingerprint
from skfp.utils.validators import ensure_mols, ensure_smiles, require_mols_with_conf_ids


def test_ensure_mols(mols_list):
    ensure_mols(mols_list)
    with pytest.raises(ValueError) as exc_info:
        ensure_mols(mols_list + [1])

    assert "Passed values must be RDKit Mol objects" in str(exc_info)


def test_ensure_mols_wrong_smiles():
    smiles_list = ["O", "O=N([O-])C1=C(CN=C1NCCSCc2ncccc2)Cc3ccccc3"]
    with pytest.raises(ValueError) as exc_info:
        ensure_mols(smiles_list)

    assert "Could not parse" in str(exc_info)
    assert "at index 1 as molecule" in str(exc_info)


def test_ensure_mols_in_fingerprint():
    smiles_list = ["O", "O=N([O-])C1=C(CN=C1NCCSCc2ncccc2)Cc3ccccc3"]
    fp = AtomPairFingerprint()
    with pytest.raises(ValueError) as exc_info:
        fp.transform(smiles_list)

    assert "Could not parse" in str(exc_info)
    assert "at index 1 as molecule" in str(exc_info)


def test_ensure_smiles(smiles_list):
    ensure_smiles(smiles_list)
    with pytest.raises(ValueError) as exc_info:
        ensure_smiles(smiles_list + [1])
    assert "Passed values must be SMILES strings" in str(exc_info)


def test_require_mols_with_conf_ids(mols_conformers_list, mols_list):
    require_mols_with_conf_ids(mols_conformers_list)
    with pytest.raises(TypeError) as exc_info:
        require_mols_with_conf_ids(mols_conformers_list + mols_list)
    assert "each must have conf_id property set" in str(exc_info)
