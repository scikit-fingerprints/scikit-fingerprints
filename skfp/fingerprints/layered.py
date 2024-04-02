from typing import Optional, Sequence, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array

from skfp.fingerprints.base import FingerprintTransformer
from skfp.validators import ensure_mols


class LayeredFingerprint(FingerprintTransformer):
    """Pattern fingerprint."""

    def __init__(
        self,
        fp_size: int = 2048,
        min_path: int = 1,
        max_path: int = 7,
        branched_paths: bool = True,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
    ):
        super().__init__(
            n_features_out=fp_size,
            sparse=sparse,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.fp_size = fp_size
        self.min_path = min_path
        self.max_path = max_path
        self.branched_paths = branched_paths

    def _calculate_fingerprint(
        self, X: Sequence[Union[str, Mol]]
    ) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem.rdmolops import LayeredFingerprint as RDKitLayeredFingerprint

        X = ensure_mols(X)
        X = [
            RDKitLayeredFingerprint(
                x,
                fpSize=self.fp_size,
                minPath=self.min_path,
                maxPath=self.max_path,
                branchedPaths=self.branched_paths,
            )
            for x in X
        ]
        return csr_array(X) if self.sparse else np.array(X)
