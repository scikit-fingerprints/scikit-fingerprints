from collections import Counter
from typing import Optional

import numpy as np
from rdkit.Chem import GetDistanceMatrix, Mol


def wiener_index(mol: Mol, distance_matrix: Optional[np.ndarray] = None) -> int:
    """
    Wiener Index (W).

    The implementation uses RDKit to calculate the Weiner Index for molecular graphs.
    The Weiner Index is computed as the sum of all pairwise distances in the molecular
    graph's distance matrix.

    Parameters
    ----------
    mol : RDKit Mol object
        The molecule for which the Wiener index is to be calculated.

    distance_matrix : np.ndarray, optional
        Precomputed distance matrix. If not provided, it will be calculated using RDKit.

    References
    ----------
    .. [1] `Wiener, Harry.
        "Structural determination of paraffin boiling points."
        Journal of the American chemical society 69.1 (1947): 17-20.
        <https://pubs.acs.org/doi/10.1021/ja01193a005>`_

    .. [2] `Rouvray, Dennis H.
        "Chapter 2-the rich legacy of half a century of the wiener index."
        Topology in Chemistry: 16-37.
        <https://www.sciencedirect.com/science/article/abs/pii/B9781898563761500068>`_

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.descriptors.topological import wiener_index
    >>> mol = MolFromSmiles("C1=CC=CC=C1")  # Benzene
    >>> wiener_index(mol)
    27
    """
    if distance_matrix is None:
        distance_matrix = GetDistanceMatrix(mol)
    return np.sum(distance_matrix) // 2


def average_wiener_index(
    mol: Mol, distance_matrix: Optional[np.ndarray] = None
) -> float:
    """
    Average Wiener Index (AW).

    The implementation uses RDKit to calculate the Average Wiener Index for molecular graphs.
    It is derived by dividing the Wiener index by the total number of atom pairs in the molecule,
    making it independent of the molecule size.

    Parameters
    ----------
    mol : RDKit Mol object
        The molecule for which the Average Wiener index is to be calculated.

    distance_matrix : np.ndarray, optional
        Precomputed distance matrix. If not provided, it will be calculated using RDKit.

    References
    ----------
    .. [1] `Andrey A. Dobrynin, Ivan Gutman,
        "The average Wiener index of hexagonal chains",
        Computers & Chemistry, Volume 23, Issue 6, 1999, Pages 571-576, ISSN 0097-8485,
        <https://www.sciencedirect.com/science/article/pii/S0097848599000352>`_

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.descriptors.topological import average_wiener_index
    >>> mol = MolFromSmiles("C1=CC=CC=C1")  # Benzene
    >>> average_wiener_index(mol)
    1.8
    """

    if distance_matrix is None:
        distance_matrix = GetDistanceMatrix(mol)
    wiener_idx = wiener_index(mol, distance_matrix)
    num_atoms = mol.GetNumAtoms()
    return (2 * wiener_idx) / (num_atoms * (num_atoms - 1))


def graph_distance_index(mol: Mol, distance_matrix: Optional[np.ndarray] = None) -> int:
    """
    Graph Distance Index (Tigdi).

    The implementation calculates the Graph Distance Index for molecular graphs.
    The GDI is a topological descriptor defined as the squared sum of all graph distance
    counts in the molecular graph's distance matrix. The formula for calculating GDI is given as:

    .. math::

        GDI = \\sum_{k=1}^{D} \\left(k \\cdot f_k\right)^2

    where:

    - :math:`D` is the topological diameter of the graph (the largest graph distance).
    - :math:`f_k` is the total number of distances in the graph equal to :math:`k`.

    Parameters
    ----------
    mol : RDKit Mol object
        The molecule for which the Graph Distance Index is to be calculated.

    distance_matrix : np.ndarray, optional
        Precomputed distance matrix. If not provided, it will be calculated using RDKit.

    References
    ----------
    .. [1] `Konstantinova, Elena V.
        "The discrimination ability of some topological and information
        distance indices for graphs of unbranched hexagonal systems."
        Journal of chemical information and computer sciences 36.1 (1996): 54-57.
        <https://pubs.acs.org/doi/10.1021/ci9502461>`_

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.descriptors.topological import graph_distance_index
    >>> mol = MolFromSmiles("C1=CC=CC=C1")  # Benzene
    >>> graph_distance_index(mol)
    261
    """
    if distance_matrix is None:
        distance_matrix = GetDistanceMatrix(mol)
    distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
    distance_counts = Counter(distances)
    return sum((k * f) ** 2 for k, f in distance_counts.items())


def zagreb_index(mol: Mol) -> int:
    """
    Zagreb Index.

    The implementation uses RDKit to calculate the first Zagreb index for molecular graphs.
    It is defined as the sum of the squares of the degrees of all atoms in the molecule.

    Parameters
    ----------
    mol : RDKit Mol object
        The molecule for which the Zagreb index is to be calculated.

    adjacency_matrix : np.ndarray, optional
        Precomputed adjacency matrix. If not provided, it will be calculated using RDKit.

    References
    ----------
    .. [1] `Gutman, Ivan.
        "Degree-based topological indices."
        Croatica Chemica Acta 86.4 (2013): 352.
        <https://hrcak.srce.hr/file/166451>`_

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.descriptors.topological import zagreb_index
    >>> mol = MolFromSmiles("C1=CC=CC=C1")  # Benzene
    >>> zagreb_index(mol)
    24
    """
    return sum(atom.GetDegree() ** 2 for atom in mol.GetAtoms())


def polarity_number(
    mol: Mol, distance_matrix: Optional[np.ndarray] = None, carbon_only: bool = False
) -> int:
    """
    Polarity Number (Pol).

    The implementation calculates the Polarity Number of a molecule.
    It is defined as the total number of unordered pairs of vertices (atoms)
    in a molecular graph that are separated by a graph distance of exactly 3.

    The polarity number provides information about the structural connectivity of a molecule.

    Parameters
    ----------
    mol : RDKit Mol object
        The molecule for which the Polarity Number is to be calculated.

    distance_matrix : np.ndarray, optional
        Precomputed distance matrix. If not provided, it will be calculated using RDKit.

    carbon_only : bool, default=False
        Whether to consider only carbon-carbon distances. If `True`, the distance matrix will be filtered
        to include only rows and columns corresponding to carbon atoms.

    References
    ----------
    .. [1] `Liu, Muhuo, and Bolian Liu.
        "On the Wiener polarity index."
        MATCH Commun. Math. Comput. Chem 66.1 (2011): 293-304.
        <https://match.pmf.kg.ac.rs/electronic_versions/Match66/n1/match66n1_293-304.pdf>`_

    .. [2] `Imran, Muhammad, Mehar Ali Malik, and Ramsha Javed.
        "Wiener polarity index and related molecular topological descriptors of titanium oxide nanotubes."
        International Journal of Quantum Chemistry 121.11 (2021): e26627.
        <https://onlinelibrary.wiley.com/doi/abs/10.1002/qua.26627>`_

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.descriptors.topological import polarity_number
    >>> mol = MolFromSmiles("C1=CC=CC=C1")  # Benzene
    >>> polarity_number(mol)
    3
    """
    if distance_matrix is None:
        distance_matrix = GetDistanceMatrix(mol)

    if carbon_only:
        atom_indices = [
            i for i, atom in enumerate(mol.GetAtoms()) if atom.GetSymbol() == "C"
        ]

        if not atom_indices:
            raise ValueError(
                "The molecule contains no carbon atoms, so carbon-only filtering is not possible."
            )

        distance_matrix = distance_matrix[np.ix_(atom_indices, atom_indices)]

    return (distance_matrix == 3).sum() // 2
