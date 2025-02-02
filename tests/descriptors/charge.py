import pytest

from skfp.descriptors import atomic_partial_charges
from skfp.preprocessing import MolFromSmilesTransformer


@pytest.fixture
def gasteiger_allowed_mols(mols_list):
    # Gasteiger partial charge model does not work for metals
    # allowed elements: https://github.com/rdkit/rdkit/blob/master/Code/GraphMol/PartialCharges/GasteigerParams.cpp
    # fmt: off
    allowed_elements = {
        "H", "C", "N", "O", "F", "Cl", "Br", "I", "S", "P", "Si", "B", "Be", "Mg", "Al",
    }
    # fmt: on
    return [
        mol
        for mol in mols_list
        if all(atom.GetSymbol() in allowed_elements for atom in mol.GetAtoms())
    ]


def test_atomic_partial_charges_formal(mols_list, gasteiger_allowed_mols):
    # should not throw errors
    [atomic_partial_charges(mol, partial_charge_model="formal") for mol in mols_list]


def test_atomic_partial_charges_gasteiger(mols_list, gasteiger_allowed_mols):
    # should not throw errors
    [
        atomic_partial_charges(mol, partial_charge_model="Gasteiger")
        for mol in gasteiger_allowed_mols
    ]

    with pytest.raises(ValueError, match="Failed to compute at least one atom.*"):
        [
            atomic_partial_charges(mol, partial_charge_model="Gasteiger")
            for mol in mols_list
        ]


def test_atomic_partial_charges_ignore_error():
    organometallics = [
        "CCCC[Li]",
        "CC[Zn]CC",
        "C[Al](C)C",
        "CCCC[SnH](CCCC)CCCC",
    ]
    mols = MolFromSmilesTransformer().transform(organometallics)

    with pytest.raises(ValueError, match="Failed to compute at least one atom.*"):
        [atomic_partial_charges(mol, partial_charge_model="Gasteiger") for mol in mols]

    # should not throw errors
    [
        atomic_partial_charges(
            mol, partial_charge_model="Gasteiger", charge_errors="ignore"
        )
        for mol in mols
    ]
    [
        atomic_partial_charges(
            mol, partial_charge_model="Gasteiger", charge_errors="zero"
        )
        for mol in mols
    ]
