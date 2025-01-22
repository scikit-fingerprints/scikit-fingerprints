from rdkit.Chem import Mol
from rdkit.Chem.GraphDescriptors import Kappa1, Kappa2, Kappa3

from skfp.utils.validators import validate_molecule


@validate_molecule
def kappa_shape_indices(mol: Mol) -> list[float]:
    """
    Kappa shape indices.

    Compute the first, second, and third kappa shape indices [1]_.

    Parameters
    ----------
    mol : RDKit Mol object
        The molecule for which the Kappa shape indices are to be computed.

    References
    ----------
    .. [1] `Milan, Randić
        "Novel Shape Descriptors for Molecular Graphs"
        Journal of Chemical Information and Computer Sciences 41.3 (2001):  607–613.
        <https://pubs.acs.org/doi/10.1021/ci0001031>`_

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.descriptors.kappa_shape import kappa_shape_indices
    >>> mol = MolFromSmiles("C1=CC=CC=C1")  # Benzene
    >>> kappa_shape_indices(mol)
    [3.4115708812260532, 1.6057694396735218, 0.5823992601400448]
    """
    return [Kappa1(mol), Kappa2(mol), Kappa3(mol)]
