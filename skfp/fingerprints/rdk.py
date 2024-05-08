from collections.abc import Sequence
from numbers import Integral
from typing import Optional, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from sklearn.utils._param_validation import Interval, InvalidParameterError

from skfp.bases import BaseFingerprintTransformer
from skfp.validators import ensure_mols


class RDKitFingerprint(BaseFingerprintTransformer):
    """RDKit fingerprint."""

    _parameter_constraints: dict = {
        **BaseFingerprintTransformer._parameter_constraints,
        "fp_size": [Interval(Integral, 1, None, closed="left")],
        "min_path": [Interval(Integral, 1, None, closed="left")],
        "max_path": [Interval(Integral, 1, None, closed="left")],
        "use_hs": ["boolean"],
        "use_bond_order": ["boolean"],
        "num_bits_per_feature": [Interval(Integral, 1, None, closed="left")],
    }

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
        self.min_path = min_path
        self.max_path = max_path
        self.use_hs = use_hs
        self.use_bond_order = use_bond_order
        self.num_bits_per_feature = num_bits_per_feature

    def _validate_params(self) -> None:
        super()._validate_params()
        if self.max_path < self.min_path:
            raise InvalidParameterError(
                f"The max_distance parameter of {self.__class__.__name__} must be "
                f"greater or equal to min_distance, got: "
                f"min_distance={self.min_path}, max_distance={self.max_path}"
            )

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
            X = [gen.GetCountFingerprintAsNumPy(mol) for mol in X]
        else:
            X = [gen.GetFingerprintAsNumPy(mol) for mol in X]

        dtype = np.uint32 if self.count else np.uint8
        if self.sparse:
            return csr_array(X, dtype=dtype)
        else:
            return np.array(X, dtype=dtype)
