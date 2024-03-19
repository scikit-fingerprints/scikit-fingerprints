from typing import List, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_array

from skfp.fingerprints.base import FingerprintTransformer


class PhysiochemicalPropertiesFingerprint(FingerprintTransformer):
    def __init__(
        self,
        fp_size: int = 2048,
        variant: str = "BP",
        count: bool = False,
        sparse: bool = False,
        n_jobs: int = None,
        verbose: int = 0,
    ):
        if variant not in ["BP", "BT"]:
            raise ValueError("Variant must be one of: 'BP', 'BT'")

        super().__init__(
            count=count,
            sparse=sparse,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.fp_size = fp_size
        self.variant = variant

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray, List[str]]
    ) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem.AtomPairs.Sheridan import GetBPFingerprint, GetBTFingerprint

        X = self._validate_input(X)

        if self.variant == "BP":
            X = [GetBPFingerprint(mol) for mol in X]
        else:  # "BT" variant
            X = [GetBTFingerprint(mol) for mol in X]

        return self._hash_fingerprint_bits(X)
