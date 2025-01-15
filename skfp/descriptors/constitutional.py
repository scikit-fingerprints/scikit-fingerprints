from rdkit.Chem import AddHs, Mol
from rdkit.Chem.Descriptors import MolWt
from rdkit.Chem.rdMolDescriptors import CalcNumRings, CalcNumRotatableBonds


def atom_count(mol: Mol, atom_symbol: str) -> int:
    """
    Specific Atom Count.

    Calculates the count of atoms [1]_ of a specific type (e.g., "C" for carbon, "H" for hydrogen).

    Parameters
    ----------
    mol : RDKit Mol object
        The molecule for which the atom count is to be calculated.

    atom_symbol : str
        The symbol of the atom type to count (e.g., "H" for hydrogen, "C" for carbon).

    References
    ----------
    .. [1] `Gerta, Rucker and Christoph, Ruecker.
        "Counts of All Walks as Atomic and Molecular Descriptors"
        Journal of Chemical Information and Computer Sciences 33.5 (1993): 683–695.
        <https://doi.org/10.1021/ci00015a005>`_

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.descriptors.constitutional import atom_count
    >>> mol = MolFromSmiles("C1=CC=CC=C1")  # Benzene
    >>> atom_count(mol, "C")
    6
    """
    mol_with_h = AddHs(mol)
    return sum(1 for atom in mol_with_h.GetAtoms() if atom.GetSymbol() == atom_symbol)


def average_molecular_weight(mol: Mol) -> float:
    """
    Average Molecular Weight.

    Calculates the average molecular weight of the molecule [1]_, defined as the molecular
    weight divided by the number of atoms.

    Parameters
    ----------
    mol : RDKit Mol object
        The molecule for which the average molecular weight is to be calculated.

    References
    ----------
    .. [1] `Rinta, Kawagoe.
        "Exploring Molecular Descriptors and Acquisition Functions in Bayesian
        Optimization for Designing Molecules with Low Hole Reorganization Energy"
        ACS Omega 9.49 (2024): 48844–48854.
        <https://doi.org/10.1021/acsomega.4c09124>`_

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.descriptors.constitutional import average_molecular_weight
    >>> mol = MolFromSmiles("C1=CC=CC=C1")  # Benzene
    >>> average_molecular_weight(mol)
    13.019
    """
    num_atoms = mol.GetNumAtoms()
    if num_atoms == 0:
        raise ValueError(
            "The molecule has no atoms, average molecular weight cannot be calculated."
        )
    return MolWt(mol) / num_atoms


def molecular_weight(mol: Mol) -> float:
    """
    Molecular Weight.

    Calculates the molecular weight of the molecule [1]_.

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

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.descriptors.constitutional import molecular_weight
    >>> mol = MolFromSmiles("C1=CC=CC=C1")  # Benzene
    >>> molecular_weight(mol)
    78.114
    """
    return MolWt(mol)


def number_of_double_bonds(mol: Mol) -> int:
    """
    Number of Double Bonds.

    Calculates the total number of double bonds in the molecule [1]_.

    Parameters
    ----------
    mol : RDKit Mol object
        The molecule for which the number of double bonds is to be calculated.

    References
    ----------
    .. [1] `Jesús, Sánchez-Márquez.
        "Itroducing new reactivity descriptors: “Bond reactivity indices.”
        Comparison of the new definitions and atomic reactivity indices"
        The Journal of Chemical Physics 145 (2016): 194105.
        <https://pubs.aip.org/aip/jcp/article-abstract/145/19/194105/932003/Introducing-new-reactivity-descriptors-Bond?redirectedFrom=fulltext>`_

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.descriptors.constitutional import number_of_double_bonds
    >>> mol = MolFromSmiles("C=CC=C")  # Butadiene
    >>> number_of_double_bonds(mol)
    2
    """
    return sum(1 for bond in mol.GetBonds() if bond.GetBondType().name == "DOUBLE")


def number_of_rings(mol: Mol) -> int:
    """
    Number of Rings.

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


def number_of_rotatable_bonds(mol: Mol) -> int:
    """
    Number of Rotatable Bonds.

    Calculates the total number of rotatable bonds in the molecule [1]_.

    Parameters
    ----------
    mol : RDKit Mol object
        The molecule for which the number of rotatable bonds is to be calculated.

    References
    ----------
    .. [1] `Jessica, Braun.
        "Understanding and Quantifying Molecular Flexibility: Torsion Angular Bin Strings"
        Journal of Chemical Information and Modeling 64.20 (2024): 7917–7924.
        <https://doi.org/10.1021/acs.jcim.4c01513>`_

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.descriptors.constitutional import number_of_rotatable_bonds
    >>> mol = MolFromSmiles("C=CC=C")  # Butadiene
    >>> number_of_rotatable_bonds(mol)
    1
    """
    return CalcNumRotatableBonds(mol)


def number_of_single_bonds(mol: Mol) -> int:
    """
    Number of Single Bonds.

    Calculates the total number of single bonds in the molecule [1]_.

    Parameters
    ----------
    mol : RDKit Mol object
        The molecule for which the number of single bonds is to be calculated.

    References
    ----------
    .. [1] `Wojciech, Grochala.
        "A focus on penetration index – a new descriptor of chemical bonding"
        Royal Society of Chemistry 14 (2023): 11597.
        <https://pubs.rsc.org/en/content/articlepdf/2023/sc/d3sc90191b>`_

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.descriptors.constitutional import number_of_single_bonds
    >>> mol = MolFromSmiles("CCO")  # Ethanol
    >>> number_of_single_bonds(mol)
    2
    """
    return sum(1 for bond in mol.GetBonds() if bond.GetBondType().name == "SINGLE")


def number_of_triple_bonds(mol: Mol) -> int:
    """
    Number of Triple Bonds.

    Calculates the total number of triple bonds in the molecule [1]_.

    Parameters
    ----------
    mol : RDKit Mol object
        The molecule for which the number of triple bonds is to be calculated.

    References
    ----------
    .. [1] `Lu, T. Xu.
        "Variations in the Nature of Triple Bonds: The N2, HCN, and HC2H Series"
        The Journal of Physical Chemistry A 120.26 (2016): 4526–4533.
        <https://pubs.acs.org/doi/10.1021/acs.jpca.6b03631>`_

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.descriptors.constitutional import number_of_triple_bonds
    >>> mol = MolFromSmiles("C#N")  # Hydrogen cyanide
    >>> number_of_triple_bonds(mol)
    1
    """
    return sum(1 for bond in mol.GetBonds() if bond.GetBondType().name == "TRIPLE")


def total_atom_count(mol: Mol) -> int:
    """
    Total Atom Count.

    Calculates the total number of atoms in the molecule [1]_.

    Parameters
    ----------
    mol : RDKit Mol object
        The molecule for which the total atom count is to be calculated.

    References
    ----------
    .. [1] `Gerta, Rucker and Christoph, Ruecker.
        "Counts of All Walks as Atomic and Molecular Descriptors"
        Journal of Chemical Information and Computer Sciences 33.5 (1993): 683–695.
        <https://doi.org/10.1021/ci00015a005>`_

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
