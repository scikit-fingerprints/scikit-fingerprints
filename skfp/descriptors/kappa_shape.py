from typing import List

from rdkit.Chem import Mol
from rdkit.Chem.GraphDescriptors import Kappa1, Kappa2, Kappa3

from skfp.utils.validators import validate_molecule


@validate_molecule
def kappa_shape_indices(mol: Mol) -> list[float]:
    """
    Compute the first, second, and third kappa shape indices.

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
    [3.412, 1.606, 0.582]
    """
    return [Kappa1(mol), Kappa2(mol), Kappa3(mol)]
