import pytest

from skfp.utils.validators import ensure_mols, ensure_smiles, require_mols_with_conf_ids


def test_ensure_mols(mols_list):
    ensure_mols(mols_list)
    with pytest.raises(ValueError) as exc_info:
        ensure_mols(mols_list + [1])
    assert "either rdkit.Chem.rdChem.Mol or SMILES" in str(exc_info)


def test_ensure_smiles(smiles_list):
    ensure_smiles(smiles_list)
    with pytest.raises(ValueError) as exc_info:
        ensure_smiles(smiles_list + [1])
    assert "Passed values must be SMILES strings" in str(exc_info)


def test_require_mols_with_conf_ids(mols_conformers_list, mols_list):
    require_mols_with_conf_ids(mols_conformers_list)
    with pytest.raises(ValueError) as exc_info:
        require_mols_with_conf_ids(mols_conformers_list + mols_list)
    assert "each must have conf_id property set" in str(exc_info)
