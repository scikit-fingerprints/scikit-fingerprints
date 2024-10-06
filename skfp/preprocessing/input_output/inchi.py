from collections.abc import Sequence

from rdkit.Chem import Mol, MolFromInchi, MolToInchi

from skfp.bases import BasePreprocessor
from skfp.utils.validators import check_mols, check_strings


class MolFromInchiTransformer(BasePreprocessor):
    """
    Creates RDKit ``Mol`` objects from InChI (International Chemical Identifier)
    strings [1]_.

    Parameters
    ----------
    sanitize : bool, default=True
        Whether to perform sanitization [1]_, i.e. basic validity checks, on created
        molecules.

    remove_hydrogens : bool, default=True
        Remove explicit hydrogens from the molecule where possible, using RDKit
        implicit hydrogens instead.

    References
    ----------
    .. [1] `RDKit InChI documentation
        <https://www.rdkit.org/docs/source/rdkit.Chem.inchi.html>`_

    Examples
    --------
    >>> from skfp.preprocessing import MolFromInchiTransformer
    >>> inchi_list = ["1S/H2O/h1H2", "1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(6)2/h4H,1-3H3"]
    >>> mol_from_inchi = MolFromInchiTransformer()
    >>> mol_from_inchi
    MolFromInchiTransformer()

    >>> mol_from_inchi.transform(inchi_list)  # doctest: +SKIP
        [<rdkit.Chem.rdchem.Mol>,
         <rdkit.Chem.rdchem.Mol>]
    """

    _parameter_constraints: dict = {
        "sanitize": ["boolean"],
        "remove_hydrogens": ["boolean"],
    }

    def __init__(
        self,
        sanitize: bool = True,
        remove_hydrogens: bool = True,
    ):
        self.sanitize = sanitize
        self.remove_hydrogens = remove_hydrogens

    def transform(self, X: Sequence[str], copy: bool = False) -> list[Mol]:
        # no parallelization, too fast to benefit from it
        self._validate_params()
        check_strings(X)
        return [
            MolFromInchi(inchi, sanitize=self.sanitize, removeHs=self.remove_hydrogens)
            for inchi in X
        ]


class MolToInchiTransformer(BasePreprocessor):
    """
    Creates InChI (International Chemical Identifier) strings from RDKit ``Mol``
    objects [1]_.

    References
    ----------
    .. [1] `RDKit InChI documentation
        <https://www.rdkit.org/docs/source/rdkit.Chem.inchi.html>`_

    Examples
    --------
    >>> from skfp.preprocessing import MolFromInchiTransformer, MolToInchiTransformer
    >>> inchi_list = ["1S/H2O/h1H2", "1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(6)2/h4H,1-3H3"]
    >>> mol_from_inchi = MolFromInchiTransformer()
    >>> mol_to_inchi = MolToInchiTransformer()
    >>> mol_to_inchi
    MolToInchiTransformer()

    >>> mols = mol_from_inchi.transform(inchi_list)
    >>> mol_to_inchi.transform(mols)
    ["1S/H2O/h1H2", "1S/C8H10N4O2/c1-10-4-9-6-5(10)7(13)12(3)8(14)11(6)2/h4H,1-3H3"]
    """

    def transform(self, X: Sequence[Mol], copy: bool = False) -> list[str]:
        # no parallelization, too fast to benefit from it
        check_mols(X)
        return [MolToInchi(mol) for mol in X]
