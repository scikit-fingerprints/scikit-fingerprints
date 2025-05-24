from numbers import Integral

import numpy as np
from rdkit.Chem import Mol
from rdkit.Chem.rdFMCS import FindMCS
from sklearn.utils._param_validation import Interval, validate_params


@validate_params(
    {
        "mol_a": [Mol],
        "mol_b": [Mol],
        "timeout": [Interval(Integral, 0, None, closed="neither")],
    },
    prefer_skip_nested_validation=True,
)
def mcs_similarity(mol_a: Mol, mol_b: Mol, timeout: int = 3600) -> float:
    r"""
    MCS similarity between molecules.

    Computes the Maximum Common Substructure (MCS) similarity [1]_ between two
    RDKit ``Mol`` objects, using the formula:

    .. math::

        sim(mol_a, mol_b) = \frac{numAtoms(MCS(mol_a, mol_b))}
        {numAtoms(mol_a) + numAtoms(mol_b) - numAtoms(MCS(mol_a, mol_b))}

    Number of atoms in MCS measures the structural overlap between molecules.
    FMCS algorithm [2]_ [3]_ [4]_ [5]_ is used for MCS computation. This measure
    penalizes the difference in size (number of atoms) between molecules.

    The calculated similarity falls within the range :math:`[0, 1]`.

    Parameters
    ----------
    mol_a : RDKit ``Mol`` object
        First molecule.

    mol_b : RDKit ``Mol`` object
        Second molecule.

    timeout : int, default=3600
        MCS computation timeout.

    Returns
    -------
    similarity : float
        MCS similarity between ``mol_a`` and ``mol_b``.

    References
    ----------
    .. [1] `Zhang, B., Vogt, M., Maggiora, G.M. et al.
        "Design of chemical space networks using a Tanimoto similarity variant based upon maximum common substructures"
        J Comput Aided Mol Des 29, 937–950 (2015)
        <https://doi.org/10.1007/s10822-015-9872-1>`_

    .. [2] `Dalke, A., Hastings, J.
        "FMCS: a novel algorithm for the multiple MCS problem"
        J Cheminform 5 (Suppl 1), O6 (2013)
        <https://doi.org/10.1186/1758-2946-5-S1-O6>`_

    .. [3] `Dalke Scientific - MCS background
        <http://dalkescientific.com/writings/diary/archive/2012/05/12/mcs_background.html>`_

    .. [4] `RDKit documentation - FMCS module documentation
        <https://www.rdkit.org/docs/source/rdkit.Chem.fmcs.fmcs.html>`_

    .. [5] `TeachOpenCADD - Maximum common substructure
        <https://projects.volkamerlab.org/teachopencadd/talktorials/T006_compound_maximum_common_substructures.html>`_

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.distances import mcs_similarity
    >>> mol_a = MolFromSmiles("COc1cc(CN2CCC(NC(=O)c3cncc(C)c3)CC2)c(OC)c2ccccc12")
    >>> mol_b = MolFromSmiles("COc1ccccc1")
    >>> sim = mcs_similarity(mol_a, mol_b)
    >>> sim
    0.25806451612903225
    """
    mcs_num_atoms = FindMCS((mol_a, mol_b), timeout=timeout).numAtoms
    denom = mol_a.GetNumAtoms() + mol_b.GetNumAtoms() - mcs_num_atoms
    sim = mcs_num_atoms / denom
    return sim


@validate_params(
    {
        "mol_a": [Mol],
        "mol_b": [Mol],
        "timeout": [Interval(Integral, 0, None, closed="neither")],
    },
    prefer_skip_nested_validation=True,
)
def mcs_distance(mol_a: Mol, mol_b: Mol, timeout: int = 3600) -> float:
    """
    MCS distance between molecules.

    Computes the Maximum Common Substructure (MCS) distance [1]_ between two RDKit
    ``Mol`` objects by subtracting similarity value from 1, using the formula:

    .. math::

        dist(a, b) = 1 - sim(a, b)

    Number of atoms in MCS measures the structural overlap between molecules.
    FMCS algorithm [2]_ [3]_ [4]_ [5]_ is used for MCS computation. This measure
    penalizes the difference in size (number of atoms) between molecules.

    See also :py:func:`mcs_binary_similarity`.
    The calculated distance falls within the range :math:`[0, 1]`.

    Parameters
    ----------
    mol_a : RDKit ``Mol`` object
        First molecule.

    mol_b : RDKit ``Mol`` object
        Second molecule.

    timeout : int, default=3600
        MCS computation timeout.

    Returns
    -------
    similarity : float
        MCS distance between ``mol_a`` and ``mol_b``.

    References
    ----------
    .. [1] `Zhang, B., Vogt, M., Maggiora, G.M. et al.
        "Design of chemical space networks using a Tanimoto similarity variant based upon maximum common substructures"
        J Comput Aided Mol Des 29, 937–950 (2015)
        <https://doi.org/10.1007/s10822-015-9872-1>`_

    .. [2] `Dalke, A., Hastings, J.
        "FMCS: a novel algorithm for the multiple MCS problem"
        J Cheminform 5 (Suppl 1), O6 (2013)
        <https://doi.org/10.1186/1758-2946-5-S1-O6>`_

    .. [3] `Dalke Scientific - MCS background
        <http://dalkescientific.com/writings/diary/archive/2012/05/12/mcs_background.html>`_

    .. [4] `RDKit documentation - FMCS module documentation
        <https://www.rdkit.org/docs/source/rdkit.Chem.fmcs.fmcs.html>`_

    .. [5] `TeachOpenCADD - Maximum common substructure
        <https://projects.volkamerlab.org/teachopencadd/talktorials/T006_compound_maximum_common_substructures.html>`_

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.distances import mcs_distance
    >>> mol_a = MolFromSmiles("COc1cc(CN2CCC(NC(=O)c3cncc(C)c3)CC2)c(OC)c2ccccc12")
    >>> mol_b = MolFromSmiles("COc1ccccc1")
    >>> dist = mcs_distance(mol_a, mol_b)
    >>> dist
    0.7419354838709677
    """
    return 1 - mcs_similarity(mol_a, mol_b, timeout)


@validate_params(
    {
        "mol_a": [list],
        "mol_b": [list],
        "timeout": [Interval(Integral, 0, None, closed="neither")],
    },
    prefer_skip_nested_validation=True,
)
def bulk_mcs_similarity(
    X: list[Mol], Y: list[Mol] | None = None, timeout: int = 3600
) -> np.ndarray:
    r"""
    Bulk MCS similarity.

    Computes the pairwise Maximum Common Substructure (MCS) similarity between lists of molecules.
    If a single list is passed, similarities are computed between its molecules.
    For two lists, similarities are between their respective molecules, with `i`-th row
    and `j`-th column in output corresponding to `i`-th molecule from first list
    and `j`-th molecule from second list.

    See also :py:func:`mcs_similarity`.

    Parameters
    ----------
    X : ndarray
        First list of molecules, of length `m`.

    Y : ndarray, default=None
        First list of molecules, of length `n`. If not passed, similarities are
        computed between molecules from X.

    timeout : int, default=3600
        MCS computation timeout.

    Returns
    -------
    similarities : ndarray
        Array with pairwise MCS similarity values. Shape is :math:`m \times n` if two
        lists of molecules are passed, or :math:`m \times m` otherwise.

    See Also
    --------
    :py:func:`mcs_similarity` : MCS similarity function for two molecules.

    Examples
    --------
    >>> from skfp.distances import bulk_mcs_similarity
    >>> from rdkit.Chem import MolFromSmiles
    >>> mols = [MolFromSmiles("COc1ccccc1"), MolFromSmiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")]
    >>> sim = bulk_mcs_similarity(mols)
    >>> sim
    array([[1.        , 0.15789474],
           [0.15789474, 1.        ]])
    """
    if Y is None:
        Y = X

    sims = np.empty((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            sims[i, j] = mcs_similarity(X[i], Y[j], timeout)

    return sims


@validate_params(
    {
        "mol_a": [list],
        "mol_b": [list],
        "timeout": [Interval(Integral, 0, None, closed="neither")],
    },
    prefer_skip_nested_validation=True,
)
def bulk_mcs_distance(
    X: list[Mol], Y: list[Mol] | None = None, timeout: int = 3600
) -> np.ndarray:
    r"""
    Bulk MCS distance.

    Computes the pairwise Maximum Common Substructure (MCS) distance between lists of molecules.
    If a single list is passed, distances are computed between its molecules.
    For two lists, distances are between their respective molecules, with `i`-th row
    and `j`-th column in output corresponding to `i`-th molecule from first list
    and `j`-th molecule from second list.

    See also :py:func:`mcs_distance`.

    Parameters
    ----------
    X : ndarray
        First list of molecules, of length `m`.

    Y : ndarray, default=None
        First list of molecules, of length `n`. If not passed, distances are
        computed between molecules from X.

    timeout : int, default=3600
        MCS computation timeout.

    Returns
    -------
    distances : ndarray
        Array with pairwise MCS distance values. Shape is :math:`m \times n` if two
        lists of molecules are passed, or :math:`m \times m` otherwise.

    See Also
    --------
    :py:func:`mcs_distance` : MCS distance function for two molecules.

    Examples
    --------
    >>> from skfp.distances import bulk_mcs_distance
    >>> from rdkit.Chem import MolFromSmiles
    >>> mols = [MolFromSmiles("COc1ccccc1"), MolFromSmiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")]
    >>> dist = bulk_mcs_distance(mols)
    >>> dist
    array([[0.        , 0.84210526],
           [0.84210526, 0.        ]])
    """
    if Y is None:
        Y = X

    sims = np.empty((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            sims[i, j] = mcs_distance(X[i], Y[j], timeout)

    return sims
