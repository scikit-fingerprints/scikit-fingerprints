import itertools
from abc import ABC
from typing import Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem.rdchem import Mol
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm

from skfp.utils.logger import tqdm_joblib


class MolFromSmilesTransformer(ABC, TransformerMixin, BaseEstimator):
    def __init__(
        self,
        sanitize: bool = True,
        replacements: dict = {},
        n_jobs: int = None,
        verbose: int = 0,
        **kwargs
    ):
        self.sanitize = sanitize
        self.replacements = replacements
        self.n_jobs = effective_n_jobs(n_jobs)
        self.verbose = verbose

    def fit(self, X, y=None, **fit_params):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)

    def transform(self, X: list[str]):
        if self.n_jobs == 1:
            return self._calculate_fingerprint(X)
        else:
            batch_size = max(len(X) // self.n_jobs, 1)

            args = (
                X[i : i + batch_size] for i in range(0, len(X), batch_size)
            )

            if self.verbose > 0:
                total_batches = min(self.n_jobs, len(X))

                with tqdm_joblib(
                    tqdm(
                        desc="Calculating fingerprints...", total=total_batches
                    )
                ) as progress_bar:
                    results = Parallel(n_jobs=self.n_jobs)(
                        delayed(self._calculate_fingerprint)(X_sub)
                        for X_sub in args
                    )
            else:
                results = Parallel(n_jobs=self.n_jobs)(
                    delayed(self._calculate_fingerprint)(X_sub)
                    for X_sub in args
                )

            return list(itertools.chain(*results))

    def _calculate_fingerprint(self, X: Union[list[str]]) -> list[Mol]:
        return [
            MolFromSmiles(
                x, sanitize=self.sanitize, replacements=self.replacements
            )
            for x in X
        ]


class MolToSmilesTransformer(ABC, TransformerMixin, BaseEstimator):
    def __init__(
        self,
        isomeric_smiles: bool = True,
        kekule_smiles: bool = False,
        rooted_at_atom: int = -1,
        canonical: bool = True,
        all_bonds_explicit: bool = False,
        all_hs_explicit: bool = False,
        do_random: bool = False,
        n_jobs: int = None,
        verbose: int = 0,
    ):
        self.isomeric_smiles = isomeric_smiles
        self.kekule_smiles = kekule_smiles
        self.rooted_at_atom = rooted_at_atom
        self.canonical = canonical
        self.all_bonds_explicit = all_bonds_explicit
        self.all_hs_explicit = all_hs_explicit
        self.do_random = do_random
        self.n_jobs = effective_n_jobs(n_jobs)
        self.verbose = verbose

    def fit(self, X, y=None, **fit_params):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)

    def transform(self, X: list[Mol]):
        if self.n_jobs == 1:
            return self._calculate_fingerprint(X)
        else:
            batch_size = max(len(X) // self.n_jobs, 1)

            args = (
                X[i : i + batch_size] for i in range(0, len(X), batch_size)
            )

            if self.verbose > 0:
                total_batches = min(self.n_jobs, len(X))

                with tqdm_joblib(
                    tqdm(
                        desc="Calculating fingerprints...", total=total_batches
                    )
                ) as progress_bar:
                    results = Parallel(n_jobs=self.n_jobs)(
                        delayed(self._calculate_fingerprint)(X_sub)
                        for X_sub in args
                    )
            else:
                results = Parallel(n_jobs=self.n_jobs)(
                    delayed(self._calculate_fingerprint)(X_sub)
                    for X_sub in args
                )

            return list(itertools.chain(*results))

    def _calculate_fingerprint(self, X: list[Mol]) -> list[str]:
        return [
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
