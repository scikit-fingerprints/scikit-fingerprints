from collections.abc import Sequence
from typing import Any, Optional, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array

from skfp.validators import ensure_mols

from .base import FingerprintTransformer


class SubstructureFingerprint(FingerprintTransformer):
    _parameter_constraints: dict = {**FingerprintTransformer._parameter_constraints}

    def __init__(
        self,
        substructures: Sequence[str],
        count: bool = False,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        random_state: Optional[int] = 0,
    ):
        super().__init__(
            n_features_out=len(substructures),
            count=count,
            sparse=sparse,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
        )
        self.substructures = ensure_mols(substructures)

    def _calculate_fingerprint(
        self, X: Sequence[Union[str, Mol]]
    ) -> Union[np.ndarray, csr_array]:
        X = ensure_mols(X)

        if self.count:
            fps = [
                [
                    len(mol.GetSubstructMatches(substructure))
                    for substructure in self.substructures
                ]
                for mol in X
            ]
        else:
            fps = [
                [
                    mol.HasSubstructMatch(substructure)
                    for substructure in self.substructures
                ]
                for mol in X
            ]

        return csr_array(fps) if self.sparse else np.array(fps)
