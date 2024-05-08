from collections.abc import Sequence
from numbers import Integral
from typing import Optional, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from sklearn.utils._param_validation import Interval

from skfp.bases import BaseFingerprintTransformer
from skfp.validators import ensure_mols


class ECFPFingerprint(BaseFingerprintTransformer):
    """Extended Connectivity Fingerprint (ECFP) transformer."""

    _parameter_constraints: dict = {
        **BaseFingerprintTransformer._parameter_constraints,
        "fp_size": [Interval(Integral, 1, None, closed="left")],
        "radius": [Interval(Integral, 0, None, closed="left")],
        "use_fcfp": ["boolean"],
        "include_chirality": ["boolean"],
        "use_bond_types": ["boolean"],
        "only_nonzero_invariants": ["boolean"],
        "include_ring_membership": ["boolean"],
        "count_bounds": [list, None],
        "use_2D": ["boolean"],
    }

    def __init__(
        self,
        fp_size: int = 2048,
        radius: int = 3,
        use_fcfp: bool = False,
        include_chirality: bool = False,
        use_bond_types: bool = True,
        only_nonzero_invariants: bool = False,
        include_ring_membership: bool = True,
        count_bounds: Optional[list] = None,
        count: bool = False,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: int = 0,
    ):
        super().__init__(
            n_features_out=fp_size,
            count=count,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
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

    def _calculate_fingerprint(
        self, X: Sequence[Union[str, Mol]]
    ) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem.rdFingerprintGenerator import (
            GetMorganFeatureAtomInvGen,
            GetMorganGenerator,
        )

        X = ensure_mols(X)

        invgen = GetMorganFeatureAtomInvGen() if self.use_fcfp else None
        gen = GetMorganGenerator(
            radius=self.radius,
            includeChirality=self.include_chirality,
            useBondTypes=self.use_bond_types,
            onlyNonzeroInvariants=self.only_nonzero_invariants,
            includeRingMembership=self.include_ring_membership,
            countBounds=self.count_bounds,
            fpSize=self.fp_size,
            atomInvariantsGenerator=invgen,
        )

        if self.count:
            X = [gen.GetCountFingerprintAsNumPy(mol) for mol in X]
        else:
            X = [gen.GetFingerprintAsNumPy(mol) for mol in X]

        return csr_array(X) if self.sparse else np.array(X)
