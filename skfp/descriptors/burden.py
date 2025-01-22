from typing import List

from rdkit.Chem import Mol
from rdkit.Chem.rdMolDescriptors import BCUT2D

from skfp.utils.validators import validate_molecule


@validate_molecule
def burden_descriptors(mol: Mol) -> list[float]:
    """
    Burden descriptors (BCUT).

    Computes the BCUT descriptors [1]_ based on the eigenvalues of the Burden adjacency matrix
    weighted by different atomic properties.

    The function returns 8 values corresponding to:
    - Highest and lowest eigenvalues weighted by atomic mass (`bcutm1`, `bcutm2`)
    - Highest and lowest eigenvalues weighted by Gasteiger charge (`bcutc1`, `bcutc2`)
    - Highest and lowest eigenvalues weighted by Crippen logP (`bcutp1`, `bcutp2`)
    - Highest and lowest eigenvalues weighted by Crippen molar refractivity (`bcutr1`, `bcutr2`)

    Parameters
    ----------
    mol : RDKit Mol object
        The molecule for which the Burden descriptors are to be calculated.

    References
    ----------
    .. [1] `Marrs, Frank W.
        "Chemical Descriptors for a Large-Scale Study on Drop-Weight Impact Sensitivity of High Explosives"
        Journal of Chemical Information and Modeling 63.3 (2023): 753–769.
        <https://pubs.acs.org/doi/10.1021/acs.jcim.2c01154>`_

    Examples
    --------
    >>> from rdkit.Chem import MolFromSmiles
    >>> from skfp.descriptors.burden import burden_descriptors
    >>> mol = MolFromSmiles("C1=CC=CC=C1")  # Benzene
    >>> burden_descriptors(mol)
    [13.646993161855445, 10.379006838144555, 1.5737245910733615, -1.6942617326375462, 1.794093161855452, -1.4738931618554547, 4.985993161855452, 1.718006838144549]
    """
    return BCUT2D(mol)
