from typing import Union

import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem.rdchem import Mol

from skfp.fingerprints.base import FingerprintTransformer


class MolFromSmilesTransformer(FingerprintTransformer):
    def __init__(
        self,
        sanitize: bool = True,
        replacements: dict = {},
        random_state: int = 0,
        n_jobs: int = None,
        verbose: int = 0,
        **kwargs
    ):
        super().__init__(
            n_jobs=n_jobs,
            sparse=False,
            count=False,
            verbose=verbose,
            random_state=random_state,
        )
        self.sanitize = sanitize
        self.replacements = replacements
        self.fingerprint_transformer = False

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, list[str]]
    ) -> pd.Series:
        return pd.Series(
            [
                MolFromSmiles(
                    x, sanitize=self.sanitize, replacements=self.replacements
                )
                for x in X
            ]
        )


class MolToSmilesTransformer(FingerprintTransformer):
    def __init__(
        self,
        isomeric_smiles: bool = True,
        kekule_smiles: bool = False,
        rooted_at_atom: int = -1,
        canonical: bool = True,
        all_bonds_explicit: bool = False,
        all_hs_explicit: bool = False,
        do_random: bool = False,
        random_state: int = 0,
        n_jobs: int = None,
        verbose: int = 0,
    ):
        super().__init__(
            n_jobs=n_jobs,
            sparse=False,
            count=False,
            verbose=verbose,
            random_state=random_state,
        )
        self.isomeric_smiles = isomeric_smiles
        self.kekule_smiles = kekule_smiles
        self.rooted_at_atom = rooted_at_atom
        self.canonical = canonical
        self.all_bonds_explicit = all_bonds_explicit
        self.all_hs_explicit = all_hs_explicit
        self.do_random = do_random

        self.fingerprint_transformer = False

    def _calculate_fingerprint(self, X: list[Mol]) -> pd.Series:
        return pd.Series(
            [
                MolToSmiles(
                    x,
                    isomericSmiles=self.isomeric_smiles,
                    kekuleSmiles=self.kekule_smiles,
                    rootedAtAtom=self.rooted_at_atom,
                    canonical=self.canonical,
                    allBondsExplicit=self.all_bonds_explicit,
                    allHsExplicit=self.all_hs_explicit,
                    doRandom=self.do_random,
                )
                for x in X
            ]
        )
