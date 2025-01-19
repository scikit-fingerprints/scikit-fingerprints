from typing import Optional, Union

from rdkit.Chem import AddHs, Mol
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.rdMolDescriptors import CalcNumRings, CalcNumRotatableBonds

from skfp.utils.validators import validate_molecule


@validate_molecule
def average_molecular_weight(mol: Mol) -> float:
    """
    Average Molecular Weight.

    Calculates the average molecular weight of the molecule, defined as the molecular
    weight divided by the number of atoms.

    This is different from "average molecular weight" in the context of isotopes [1]_.

    Parameters
    ----------
    mol : RDKit Mol object
        The molecule for which the average molecular weight is to be calculated.

    References
    ----------
    .. [1] `
        <https://chemistry.stackexchange.com/questions/150993/discrepancy-when-calulating-mol-weights-with-chemsketch-and-python-rdkit/150999#150999>`_

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.descriptors.constitutional import average_molecular_weight
    >>> mol = MolFromSmiles("C1=CC=CC=C1")  # Benzene
    >>> average_molecular_weight(mol)
    13.019
    """
    return MolWt(mol) / mol.GetNumAtoms()


@validate_molecule
def bond_type_count(mol: Mol, bond_type: Optional[str] = None) -> int:
    """
    Bond Type Count.

    Counts the total number of bonds of a specific type in the molecule [1]_.

    Parameters
    ----------
    mol : RDKit Mol object
        The molecule for which the bond count is to be calculated.

    bond_type : str, optional
        Valid options are RDKit bond types [2]_:
        - "SINGLE"
        - "DOUBLE"
        - "TRIPLE"
        - "AROMATIC"
        If "None", the function returns the total number of bonds.

    References
    ----------
    .. [1] `Wojciech, Grochala.
        "A focus on penetration index – a new descriptor of chemical bonding"
        Royal Society of Chemistry 14 (2023): 11597.
        <https://pubs.rsc.org/en/content/articlepdf/2023/sc/d3sc90191b>`_

    .. [2] `
        <https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.BondType>`_

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> mol = MolFromSmiles("C1=CC=CC=C1")  # Benzene
    >>> bond_type_count(mol, "AROMATIC")
    6
    >>> bond_type_count(mol, "DOUBLE")
    0
    >>> bond_type_count(mol)  # Total bonds
    6
    """
    if bond_type:
        return sum(
            1 for bond in mol.GetBonds() if bond.GetBondType().name == bond_type.upper()
        )
    return mol.GetNumBonds()


@validate_molecule
def element_atom_count(mol: Mol, atom_id: Union[int, str]) -> int:
    """
    Element atom count.

    Calculates the count of atoms of a specific type.
    In case of hydrogens, the total number is returned, counting both explicit and implicit ones.

    Parameters
    ----------
    mol : RDKit Mol object
        The molecule for which the atom count is to be calculated.

    atom_id : int or str
        The atomic number of the atom type, e.g. 6 for carbon, 1 for hydrogen
        or symbol of the atom type, e.g. "C" for carbon,  "H" for hydrogen.

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> mol = MolFromSmiles("C1=CC=CC=C1")  # Benzene
    >>> element_atom_count(mol, "C")
    6
    >>> element_atom_count(mol, 6)
    6

    >>> mol = MolFromSmiles("CCO")  # Ethanol
    >>> element_atom_count(mol, "H")
    8
    >>> element_atom_count(mol, 1)
    8
    """
    if atom_id == 1 or atom_id == "H":
        return sum(atom.GetTotalNumHs() for atom in mol.GetAtoms())
    else:
        return sum(
            1
            for atom in mol.GetAtoms()
            if atom.GetAtomicNum() == atom_id or atom.GetSymbol() == atom_id
        )


@validate_molecule
def heavy_atom_count(mol: Mol) -> int:
    """
    Heavy atom count.

    Calculates the number of heavy atoms (non-hydrogen) in the molecule.

    Parameters
    ----------
    mol : RDKit Mol object
        The molecule for which the total atom count is to be calculated.

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.descriptors.constitutional import heavy_atom_count
    >>> mol = MolFromSmiles("C1=CC=CC=C1")  # Benzene
    >>> heavy_atom_count(mol)
    6
    """
    return mol.GetNumHeavyAtoms()


@validate_molecule
def molecular_weight(mol: Mol) -> float:
    """
    Molecular Weight.

    Calculates the molecular weight of the molecule [1]_.
    This is average molecular weight in terms of isotopes [2]_.

    Parameters
    ----------
    mol : RDKit Mol object
        The molecule for which the molecular weight is to be calculated.

    References
    ----------
    .. [1] `Hiromasa, Kaneko.
        "Molecular Descriptors, Structure Generation,
        and Inverse QSAR/QSPR Based on SELFIES"
        ACS Omega 8.24 (2023): 21781–21786.
        <https://doi.org/10.1021/acsomega.3c01332>`_

    .. [2] `
        <https://chemistry.stackexchange.com/questions/150993/discrepancy-when-calulating-mol-weights-with-chemsketch-and-python-rdkit/150999#150999>`_

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.descriptors.constitutional import molecular_weight
    >>> mol = MolFromSmiles("C1=CC=CC=C1")  # Benzene
    >>> molecular_weight(mol)
    78.114
    """
    return MolWt(mol)


@validate_molecule
def number_of_rings(mol: Mol) -> int:
    """
    Number of rings.

    Calculates the total number of rings in the molecule [1]_.

    Parameters
    ----------
    mol : RDKit Mol object
        The molecule for which the number of rings is to be calculated.

    References
    ----------
    .. [1] `Alan H. Lipkus.
        "Exploring Chemical Rings in a Simple Topological-Descriptor Space"
        Journal of Chemical Information and Computer Sciences 41.2 (2001): 430–438.
        <https://doi.org/10.1021/ci000144x>`_

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.descriptors.constitutional import number_of_rings
    >>> mol = MolFromSmiles("C1=CC=CC=C1")  # Benzene
    >>> number_of_rings(mol)
    1
    """
    return CalcNumRings(mol)


@validate_molecule
def number_of_rotatable_bonds(mol: Mol) -> int:
    """
    Number of rotatable bonds.

    Calculates the total number of rotatable bonds in the molecule.

    Parameters
    ----------
    mol : RDKit Mol object
        The molecule for which the number of rotatable bonds is to be calculated.

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.descriptors.constitutional import number_of_rotatable_bonds
    >>> mol = MolFromSmiles("C=CC=C")  # Butadiene
    >>> number_of_rotatable_bonds(mol)
    1
    """
    return CalcNumRotatableBonds(mol)


@validate_molecule
def total_atom_count(mol: Mol) -> int:
    """
    Total atom count.

    Calculates the total number of atoms in the molecule.
    Includes hydrogens, both explicit and implicit.

    Parameters
    ----------
    mol : RDKit Mol object
        The molecule for which the total atom count is to be calculated.

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.descriptors.constitutional import total_atom_count
    >>> mol = MolFromSmiles("C1=CC=CC=C1")  # Benzene
    >>> total_atom_count(mol)
    12
    """
    mol_with_h = AddHs(mol)
    return mol_with_h.GetNumAtoms()
