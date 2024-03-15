from typing import Union, List

import numpy as np
import pandas as pd
from scipy.sparse import csr_array

from skfp.fingerprints.base import FingerprintTransformer


class ERGFingerprint(FingerprintTransformer):
    def __init__(
        self,
        atom_types: int = 0,
        fuzz_increment: float = 0.3,
        min_path: int = 1,
        max_path: int = 15,
        sparse: bool = False,
        n_jobs: int = None,
        verbose: int = 0,
    ):
        super().__init__(
            sparse=sparse,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.atom_types = atom_types
        self.fuzz_increment = fuzz_increment
        self.min_path = min_path
        self.max_path = max_path

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray, List[str]]
    ) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem.rdReducedGraphs import GetErGFingerprint

        X = self._validate_input(X)

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

        if self.sparse:
            return csr_array(X)
        else:
            return np.array(X)
