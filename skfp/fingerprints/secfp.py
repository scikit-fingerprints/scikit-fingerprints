from numbers import Integral
from typing import Optional, Sequence, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from sklearn.utils import Interval
from sklearn.utils._param_validation import InvalidParameterError

from skfp.validators import ensure_mols

from .base import FingerprintTransformer


class SECFPFingerprint(FingerprintTransformer):
    """SECFP fingerprint."""

    _parameter_constraints: dict = {
        **FingerprintTransformer._parameter_constraints,
        "fp_size": [Interval(Integral, 1, None, closed="left")],
        "radius": [Interval(Integral, 1, None, closed="left")],
        "min_radius": [Interval(Integral, 1, None, closed="left")],
        "rings": ["boolean"],
        "isomeric": ["boolean"],
        "kekulize": ["boolean"],
    }

    def __init__(
        self,
        fp_size: int = 2048,
        radius: int = 3,
        min_radius: int = 1,
        rings: bool = True,
        isomeric: bool = False,
        kekulize: bool = True,
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
        self.radius = radius
        self.min_radius = min_radius
        self.rings = rings
        self.isomeric = isomeric
        self.kekulize = kekulize

    def _validate_params(self) -> None:
        super()._validate_params()
        if self.radius < self.min_radius:
            raise InvalidParameterError(
                f"The radius parameter of {self.__class__.__name__} must be "
                f"greater or equal to min_radius, got: "
                f"min_radius={self.min_radius}, radius={self.radius}"
            )

    def _calculate_fingerprint(
        self, X: Sequence[Union[str, Mol]]
    ) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem.rdMHFPFingerprint import MHFPEncoder

        X = ensure_mols(X)

        # bulk function does not work
        encoder = MHFPEncoder(self.fp_size, self.random_state)
        X = [
            encoder.EncodeSECFPMol(
                x,
                length=self.fp_size,
                radius=self.radius,
                min_radius=self.min_radius,
                rings=self.rings,
                isomeric=self.isomeric,
                kekulize=self.kekulize,
            )
            for x in X
        ]

        return csr_array(X) if self.sparse else np.array(X)
