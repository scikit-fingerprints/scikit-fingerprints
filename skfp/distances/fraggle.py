from numbers import Real

from rdkit.Chem import Mol
from rdkit.Chem.Fraggle.FraggleSim import GetFraggleSimilarity
from sklearn.utils._param_validation import Interval, validate_params


@validate_params(
    {
        "mol_query": [Mol],
        "mol_ref": [Mol],
        "tversky_threshold": [Interval(Real, 0, 1, closed="both")],
    },
    prefer_skip_nested_validation=True,
)
def fraggle_similarity(
    mol_query: Mol, mol_ref: Mol, tversky_threshold: float = 0.8
) -> float:
    """
    Fraggle similarity between molecules.

    Computes the Fraggle similarity [1]_ [2]_ between two RDKit ``Mol`` objects.
    It is a hybrid between substructure and similarity search, using a "fuzzy"
    matching based on molecule fragments and Tanimoto similarity of path-based
    fingerprints.

    It is designed to properly recognize small changes in the "middle" of a
    molecule, where typical fingerprint-based measures would result in a very
    low similarity.

    This is an asymmetric measure, with query and reference molecules. It consists
    of 3 phases: query fragmentation, fragment-reference Tversky similarity comparison,
    and Tanimoto similarity comparison using RDKit fingerprints.

    Query molecule is fragmented into "interesting" substructures by acyclic and
    ring cuts, leaving only "large" parts of a molecule (>60%). Fragements are then
    compared with the reference molecule using Tversky similarity [3]_ (alpha=0.95,
    beta=0.05), keeping those with at least ``tversky_threshold`` similarity. Lastly,
    the Tanimoto similarity of RDKit fingerprints with path length 5 is computed for
    kept fragments and the reference molecule. The highest one is the Fraggle
    similarity value.

    The calculated similarity falls within the range :math:`[0, 1]`.
    Passing all-zero vectors to this function results in a similarity of 0.
    Note that this measure is asymmetric, and order of query and reference molecules
    matters.

    Parameters
    ----------
    mol_query : RDKit ``Mol`` object
        Query molecule.

    mol_ref : RDKit ``Mol`` object
        Reference molecule.

    tversky_threshold : float, default=0.8
        Required minimal Tversky similarity [3]_ between a fragment and the reference
        molecule.

    Returns
    -------
    similarity : float
        Fraggle similarity between ``mol_query`` and ``mol_ref``.

    References
    ----------
    .. [1] `Jameed Hussain, Gavin Harper
        "Fraggle – a new similarity searching algorithm"
        RDKit UGM 2013
        <https://raw.github.com/rdkit/UGM_2013/master/Presentations/Hussain.Fraggle.pdf>`_

    .. [2] `Gregory Landrum
        "Comparing Fraggle to other fingerprints"
        <https://rdkit.blogspot.com/2013/11/comparing-fraggle-to-other-fingerprints.html>`_

    .. [3] `Tversky index on Wikipedia
        <https://en.wikipedia.org/wiki/Tversky_index>`_

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.distances import fraggle_similarity
    >>> mol_query = MolFromSmiles("COc1cc(CN2CCC(NC(=O)c3cncc(C)c3)CC2)c(OC)c2ccccc12")
    >>> mol_ref = MolFromSmiles("COc1ccccc1")
    >>> sim = fraggle_similarity(mol_query, mol_ref)
    >>> sim
    0.1640625
    """
    return GetFraggleSimilarity(mol_query, mol_ref, tversky_threshold)[0]


@validate_params(
    {
        "mol_query": [Mol],
        "mol_ref": [Mol],
        "tversky_threshold": [Interval(Real, 0, 1, closed="both")],
    },
    prefer_skip_nested_validation=True,
)
def fraggle_distance(
    mol_query: Mol, mol_ref: Mol, tversky_threshold: float = 0.8
) -> float:
    """
    Fraggle distance between molecules.

    Computes the Fraggle distance [1]_ [2]_ between two RDKit ``Mol`` objects
    by subtracting similarity value from 1, using the formula:

    .. math::
        dist(a, b) = 1 - sim(a, b)

    See also :py:func:`fraggle_binary_similarity`.
    The calculated distance falls within the range :math:`[0, 1]`.
    Passing all-zero vectors to this function results in a distance of 0.
    Note that this measure is asymmetric, and order of query and reference molecules
    matters.

    Parameters
    ----------
    mol_query : RDKit ``Mol`` object
        Query molecule.

    mol_ref : RDKit ``Mol`` object
        Reference molecule.

    tversky_threshold : float, default=0.8
        Required minimal Tversky similarity [3]_ between a fragment and the reference
        molecule.

    Returns
    -------
    similarity : float
        Fraggle similarity between ``mol_query`` and ``mol_ref``.

    References
    ----------
    .. [1] `Jameed Hussain, Gavin Harper
        "Fraggle – a new similarity searching algorithm"
        RDKit UGM 2013
        <https://raw.github.com/rdkit/UGM_2013/master/Presentations/Hussain.Fraggle.pdf>`_

    .. [2] `Gregory Landrum
        "Comparing Fraggle to other fingerprints"
        <https://rdkit.blogspot.com/2013/11/comparing-fraggle-to-other-fingerprints.html>`_

    .. [3] `Tversky index on Wikipedia
        <https://en.wikipedia.org/wiki/Tversky_index>`_

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.distances import fraggle_distance
    >>> mol_query = MolFromSmiles("COc1cc(CN2CCC(NC(=O)c3cncc(C)c3)CC2)c(OC)c2ccccc12")
    >>> mol_ref = MolFromSmiles("COc1ccccc1")
    >>> dist = fraggle_distance(mol_query, mol_ref)
    >>> dist
    0.8359375
    """
    return 1 - GetFraggleSimilarity(mol_query, mol_ref, tversky_threshold)[0]
