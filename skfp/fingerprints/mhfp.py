from typing import Union, List

import numpy as np
import pandas as pd
from scipy.sparse import csr_array

from skfp.fingerprints.base import FingerprintTransformer


class MHFPFingerprint(FingerprintTransformer):
    def __init__(
        self,
        fp_size: int = 2048,
        radius: int = 3,
        min_radius: int = 1,
        rings: bool = True,
        isomeric: bool = False,
        kekulize: bool = True,
        output_raw_hashes: bool = False,
        count: bool = False,
        sparse: bool = False,
        n_jobs: int = None,
        verbose: int = 0,
    ):
        super().__init__(
            count=count,
            sparse=sparse,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.fp_size = fp_size
        self.radius = radius
        self.min_radius = min_radius
        self.rings = rings
        self.isomeric = isomeric
        self.kekulize = kekulize
        self.output_raw_hashes = output_raw_hashes

    def _calculate_fingerprint(
        self, X: Union[pd.DataFrame, np.ndarray, List[str]]
    ) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem.rdMHFPFingerprint import MHFPEncoder

        X = self._validate_input(X)

        # outputs raw hash values, not feature vectors!
        encoder = MHFPEncoder(self.fp_size, self.random_state)
        X = MHFPEncoder.EncodeMolsBulk(
            encoder,
            X,
            radius=self.radius,
            min_radius=self.min_radius,
            rings=self.rings,
            isomeric=self.isomeric,
            kekulize=self.kekulize,
        )
        X = np.array(X)

        if not self.output_raw_hashes:
            X = np.mod(X, self.fp_size)
            X = np.stack([np.bincount(x, minlength=self.fp_size) for x in X])
            if not self.count:
                X = (X > 0).astype(int)

        if self.sparse:
            return csr_array(X)
        else:
            return np.array(X)
