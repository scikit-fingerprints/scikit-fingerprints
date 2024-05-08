from collections.abc import Sequence
from numbers import Integral
from typing import Optional

from rdkit.Chem import Mol, MolFromSmiles, MolToSmiles

from skfp.bases import BasePreprocessor
from skfp.validators import ensure_mols, ensure_smiles


class MolFromSmilesTransformer(BasePreprocessor):
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
    _parameter_constraints: dict = {
        "isomeric_smiles": ["boolean"],
        "kekule_smiles": ["boolean"],
        "rooted_at_atom": [Integral],
        "canonical": ["boolean"],
        "all_bonds_explicit": ["boolean"],
        "all_hs_explicit": ["boolean"],
        "do_random": ["boolean"],
    }

    def __init__(
        self,
        isomeric_smiles: bool = True,
        kekule_smiles: bool = False,
        rooted_at_atom: int = -1,
        canonical: bool = True,
        all_bonds_explicit: bool = False,
        all_hs_explicit: bool = False,
        do_random: bool = False,
    ):
        self.isomeric_smiles = isomeric_smiles
        self.kekule_smiles = kekule_smiles
        self.rooted_at_atom = rooted_at_atom
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
                rootedAtAtom=self.rooted_at_atom,
                canonical=self.canonical,
                allBondsExplicit=self.all_bonds_explicit,
                allHsExplicit=self.all_hs_explicit,
                doRandom=self.do_random,
            )
            for mol in X
        ]
