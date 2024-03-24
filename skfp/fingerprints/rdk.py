from typing import Optional, Sequence, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array

from skfp.fingerprints.base import FingerprintTransformer
from skfp.validators import ensure_mols


class RDKitFingerprint(FingerprintTransformer):
    """RDKit fingerprint."""

    def __init__(
        self,
        fp_size: int = 2048,
        min_path: int = 1,
        max_path: int = 7,
        use_hs: bool = True,
        use_bond_order: bool = True,
        num_bits_per_feature: int = 2,
        count: bool = False,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
    ):
        super().__init__(
            count=count,
            sparse=sparse,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.fp_size = fp_size
        self.min_path = min_path
        self.max_path = max_path
        self.use_hs = use_hs
        self.use_bond_order = use_bond_order
        self.num_bits_per_feature = num_bits_per_feature

    def _calculate_fingerprint(
        self, X: Sequence[Union[str, Mol]]
    ) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem.rdFingerprintGenerator import GetRDKitFPGenerator

        X = ensure_mols(X)

        gen = GetRDKitFPGenerator(
            minPath=self.min_path,
            maxPath=self.max_path,
            useHs=self.use_hs,
            useBondOrder=self.use_bond_order,
            countSimulation=self.count,
            fpSize=self.fp_size,
            numBitsPerFeature=self.num_bits_per_feature,
        )

        if self.count:
            X = [gen.GetCountFingerprintAsNumPy(x) for x in X]
        else:
            X = [gen.GetFingerprintAsNumPy(x) for x in X]

        return csr_array(X) if self.sparse else np.array(X)
