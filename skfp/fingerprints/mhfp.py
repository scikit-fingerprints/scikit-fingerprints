from typing import Union

import numpy as np
import pandas as pd
import scipy.sparse as spsparse

from skfp.fingerprints.base import FingerprintTransformer


class MHFP(FingerprintTransformer):
    def __init__(
        self,
        n_permutations: int = 2048,
        radius: int = 3,
        rings: bool = True,
        isomeric: bool = False,
        kekulize: bool = True,
        min_radius: int = 1,
        random_state: int = 0,
        sparse: bool = False,
        count: bool = False,
        n_jobs: int = None,
        verbose: int = 0,
    ):
        self.n_permutations = n_permutations
        self.radius = radius
        self.rings = rings
        self.isomeric = isomeric
        self.kekulize = kekulize
        self.min_radius = min_radius
        super().__init__(
            n_jobs=n_jobs,
            sparse=sparse,
            count=count,
            verbose=verbose,
            random_state=random_state,
        )

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray, list[str]]
    ) -> Union[np.ndarray, spsparse.csr_array]:
        X = self._validate_input(X)
        from rdkit.Chem.rdMHFPFingerprint import MHFPEncoder

        encoder = MHFPEncoder(self.n_permutations, self.random_state)

        X = MHFPEncoder.EncodeMolsBulk(
            encoder,
            X,
            radius=self.radius,
            rings=self.rings,
            isomeric=self.isomeric,
            kekulize=self.kekulize,
            min_radius=self.min_radius,
        )

        if self.sparse:
            return spsparse.csr_array(X)
        else:
            return np.array(X)


class MHFPlegacy(FingerprintTransformer):
    def __init__(
        self,
        dimensions: int = 1024,
        radius: int = 2,
        random_state: int = 0,
        sparse: bool = False,
        count: bool = False,
        n_jobs: int = None,
        verbose: int = 0,
    ):
        super().__init__(
            n_jobs=n_jobs,
            sparse=sparse,
            count=count,
            verbose=verbose,
            random_state=random_state,
        )
        self.dimensions = dimensions
        self.radius = radius

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray, list[str]]
    ) -> Union[np.ndarray, spsparse.csr_array]:
        X = self._validate_input(X)
        from skfp.helpers.map4_mhfp_helpers import get_mhfp

        X = [
            get_mhfp(
                x,
                dimensions=self.dimensions,
                radius=self.radius,
                random_state=self.random_state,
                count=self.count,
            )
            for x in X
        ]
        if self.sparse:
            return spsparse.csr_array(X)
        else:
            return np.array(X)
