from typing import List, Optional, Union

import numpy as np
import pandas as pd
import scipy.sparse as spsparse

from skfp.fingerprints.base import FingerprintTransformer


class ECFP(FingerprintTransformer):
    def __init__(
        self,
        radius: int = 3,
        include_chirality: bool = False,
        use_bond_types: bool = True,
        only_nonzero_invariants: bool = False,
        include_ring_membership: bool = True,
        count_bounds: Optional[List] = None,
        fp_size: int = 2048,
        n_jobs: int = 1,
        sparse: bool = False,
        count: bool = False,
        verbose: int = 0,
        random_state: int = 0,
    ):
        super().__init__(
            n_jobs=n_jobs,
            sparse=sparse,
            count=count,
            verbose=verbose,
            random_state=random_state,
        )
        self.radius = radius
        self.include_chirality = include_chirality
        self.radius = radius
        self.include_chirality = include_chirality
        self.use_bond_types = use_bond_types
        self.only_nonzero_invariants = only_nonzero_invariants
        self.include_ring_membership = include_ring_membership
        self.count_bounds = count_bounds
        self.fp_size = fp_size

    def _get_generator(self):
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

        return GetMorganGenerator(
            radius=self.radius,
            includeChirality=self.include_chirality,
            useBondTypes=self.use_bond_types,
            onlyNonzeroInvariants=self.only_nonzero_invariants,
            includeRingMembership=self.include_ring_membership,
            countBounds=self.count_bounds,
            fpSize=self.fp_size,
        )

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray, list[str]]
    ) -> Union[np.ndarray, spsparse.csr_array]:
        X = self._validate_input(X)
        return self._generate_fingerprints(X)
