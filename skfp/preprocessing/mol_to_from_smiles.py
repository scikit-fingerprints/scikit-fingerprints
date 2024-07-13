from collections.abc import Sequence
from typing import Optional

from rdkit.Chem import Mol, MolFromSmiles, MolToSmiles

from skfp.bases import BasePreprocessor
from skfp.utils import ensure_mols, ensure_smiles


class MolFromSmilesTransformer(BasePreprocessor):
    """
    Creates RDKit `Mol` objects from SMILES strings.

    Parameters
    ----------
    sanitize : bool, default=True
        Whether to perform sanitization [1]_, i.e. basic validity checks, on created
        molecules.

    replacements : dict, default=None
        If provided, will be used to do string substitution of abbreviations in the
        input SMILES.

    References
    ----------
    .. [1] `Gregory Landrum
        "The RDKit Book: Molecular Sanitization"
        <https://www.rdkit.org/docs/RDKit_Book.html#molecular-sanitization>`_

    Examples
    --------
    >>> from skfp.preprocessing import MolFromSmilesTransformer
    >>> smiles = ["O", "CC", "[C-]#N", "CC=O"]
    >>> mol_from_smiles = MolFromSmilesTransformer()
    >>> mol_from_smiles
    MolFromSmilesTransformer()

    >>> mol_from_smiles.transform(smiles)  # doctest: +SKIP
        [<rdkit.Chem.rdchem.Mol object at ...>,
         <rdkit.Chem.rdchem.Mol object at ...>,
         <rdkit.Chem.rdchem.Mol object at ...>,
         <rdkit.Chem.rdchem.Mol object at ...>]
    """

    _parameter_constraints: dict = {
        "sanitize": ["boolean"],
        "replacements": [dict, None],
    }

    def __init__(
        self,
        sanitize: bool = True,
        replacements: Optional[dict] = None,
    ):
        self.sanitize = sanitize
        self.replacements = replacements

    def transform(self, X: Sequence[str], copy: bool = False) -> list[Mol]:
        # no parallelization, too fast to benefit from it
        self._validate_params()
        X = ensure_smiles(X)
        replacements = self.replacements if self.replacements else {}
        return [
            MolFromSmiles(smi, sanitize=self.sanitize, replacements=replacements)
            for smi in X
        ]


class MolToSmilesTransformer(BasePreprocessor):
    """
    Creates SMILES strings from RDKit `Mol` objects.

    Parameters
    ----------
    isomeric_smiles : bool, default=True
        Whether to include information about stereochemistry.

    kekule_smiles : bool, default=False
        Whether to use the Kekule form (no aromatic bonds).

    canonical : bool, default=True
        Whether to canonicalize the molecule. This results in a reproducible
        SMILES, given the same input molecule (if `do_random` is not used).

    all_bonds_explicit : bool, default=False
        Whether to explicitly indicate all bond orders.

    all_hs_explicit : bool, default=False
        Whether to explicitly indicate all hydrogens.

    do_random : bool, default=False
        If True, randomizes the traversal of the molecule graph, generating
        random SMILES.

    References
    ----------
    .. [1] `Gregory Landrum
        "The RDKit Book: Molecular Sanitization"
        <https://www.rdkit.org/docs/RDKit_Book.html#molecular-sanitization>`_

    Examples
    --------
    >>> from skfp.preprocessing import MolFromSmilesTransformer, MolToSmilesTransformer
    >>> smiles = ["O", "CC", "[C-]#N", "CC=O"]
    >>> mol_from_smiles = MolFromSmilesTransformer()
    >>> mol_to_smiles = MolToSmilesTransformer()
    >>> mol_to_smiles
    MolToSmilesTransformer()

    >>> mols = mol_from_smiles.transform(smiles)
    >>> mol_to_smiles.transform(mols)
    ['O', 'CC', '[C-]#N', 'CC=O']
    """

    _parameter_constraints: dict = {
        "isomeric_smiles": ["boolean"],
        "kekule_smiles": ["boolean"],
        "canonical": ["boolean"],
        "all_bonds_explicit": ["boolean"],
        "all_hs_explicit": ["boolean"],
        "do_random": ["boolean"],
    }

    def __init__(
        self,
        isomeric_smiles: bool = True,
        kekule_smiles: bool = False,
        canonical: bool = True,
        all_bonds_explicit: bool = False,
        all_hs_explicit: bool = False,
        do_random: bool = False,
    ):
        self.isomeric_smiles = isomeric_smiles
        self.kekule_smiles = kekule_smiles
        self.canonical = canonical
        self.all_bonds_explicit = all_bonds_explicit
        self.all_hs_explicit = all_hs_explicit
        self.do_random = do_random

    def transform(self, X: Sequence[Mol], copy: bool = False) -> list[str]:
        # no parallelization, too fast to benefit from it
        self._validate_params()
        X = ensure_mols(X)
        return [
            MolToSmiles(
                mol,
                isomericSmiles=self.isomeric_smiles,
                kekuleSmiles=self.kekule_smiles,
                canonical=self.canonical,
                allBondsExplicit=self.all_bonds_explicit,
                allHsExplicit=self.all_hs_explicit,
                doRandom=self.do_random,
            )
            for mol in X
        ]
