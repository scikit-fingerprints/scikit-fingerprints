from typing import List, Optional, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_array

from skfp.fingerprints.base import FingerprintTransformer


class ECFPFingerprint(FingerprintTransformer):
    def __init__(
        self,
        fp_size: int = 2048,
        radius: int = 3,
        use_fcfp: bool = False,
        include_chirality: bool = False,
        use_bond_types: bool = True,
        only_nonzero_invariants: bool = False,
        include_ring_membership: bool = True,
        count_bounds: Optional[List] = None,
        sparse: bool = False,
        count: bool = False,
        n_jobs: int = None,
        verbose: int = 0,
        random_state: int = 0,
    ):
        super().__init__(
            sparse=sparse,
            count=count,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
        )
        self.fp_size = fp_size
        self.radius = radius
        self.use_fcfp = use_fcfp
        self.include_chirality = include_chirality
        self.radius = radius
        self.include_chirality = include_chirality
        self.use_bond_types = use_bond_types
        self.only_nonzero_invariants = only_nonzero_invariants
        self.include_ring_membership = include_ring_membership
        self.count_bounds = count_bounds

    def _get_generator(self):
        from rdkit.Chem.rdFingerprintGenerator import (
            GetMorganFeatureAtomInvGen,
            GetMorganGenerator,
        )

        invgen = GetMorganFeatureAtomInvGen() if self.use_fcfp else None

        return GetMorganGenerator(
            radius=self.radius,
            includeChirality=self.include_chirality,
            useBondTypes=self.use_bond_types,
            onlyNonzeroInvariants=self.only_nonzero_invariants,
            includeRingMembership=self.include_ring_membership,
            countBounds=self.count_bounds,
            fpSize=self.fp_size,
            atomInvariantsGenerator=invgen,
        )

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray, List[str]]
    ) -> Union[np.ndarray, csr_array]:
        X = self._validate_input(X)
        return self._generate_fingerprints(X)
