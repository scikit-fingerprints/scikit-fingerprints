from typing import Optional, Sequence, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array

from skfp.fingerprints.base import FingerprintTransformer
from skfp.validators import ensure_mols


class ERGFingerprint(FingerprintTransformer):
    """Extended Reduced Graph Fingerprint (ERG) transformer."""

    def __init__(
        self,
        atom_types: int = 0,
        fuzz_increment: float = 0.3,
        min_path: int = 1,
        max_path: int = 15,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
    ):
        super().__init__(
            n_features_out=315,
            sparse=sparse,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.atom_types = atom_types
        self.fuzz_increment = fuzz_increment
        self.min_path = min_path
        self.max_path = max_path

    def _calculate_fingerprint(
        self, X: Sequence[Union[str, Mol]]
    ) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem.rdReducedGraphs import GetErGFingerprint

        X = ensure_mols(X)

        X = [
            GetErGFingerprint(
                x,
                atomTypes=self.atom_types,
                fuzzIncrement=self.fuzz_increment,
                minPath=self.min_path,
                maxPath=self.max_path,
            )
            for x in X
        ]

        return csr_array(X) if self.sparse else np.array(X)
