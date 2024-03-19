from typing import Optional, Sequence, Union

import numpy as np
from rdkit.Chem import Mol
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
        variant: str = "bit",
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
    ):
        if variant not in ["bit", "count", "raw_hashes"]:
            raise ValueError("Variant must be one of: 'bit', 'count', 'raw_hashes'")

        super().__init__(
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
        self.variant = variant

    def _calculate_fingerprint(
        self, X: Sequence[Union[str, Mol]]
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

        if self.variant in ["bit", "count"]:
            X = np.mod(X, self.fp_size)
            X = np.stack([np.bincount(x, minlength=self.fp_size) for x in X])
            if self.variant == "bit":
                X = (X > 0).astype(int)

        return csr_array(X) if self.sparse else np.array(X)
