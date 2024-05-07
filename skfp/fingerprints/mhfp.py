from collections.abc import Sequence
from numbers import Integral
from typing import Optional, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from sklearn.utils._param_validation import Interval, InvalidParameterError, StrOptions

from skfp.bases import BaseFingerprintTransformer
from skfp.validators import ensure_mols


class MHFPFingerprint(BaseFingerprintTransformer):
    """MinHashed FingerPrint (MHFP) transformer."""

    _parameter_constraints: dict = {
        **BaseFingerprintTransformer._parameter_constraints,
        "fp_size": [Interval(Integral, 1, None, closed="left")],
        "radius": [Interval(Integral, 0, None, closed="left")],
        "min_radius": [Interval(Integral, 0, None, closed="left")],
        "rings": ["boolean"],
        "isomeric": ["boolean"],
        "kekulize": ["boolean"],
        "variant": [StrOptions({"bit", "count", "raw_hashes"})],
    }

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
        batch_size: Optional[int] = None,
        verbose: int = 0,
    ):
        super().__init__(
            n_features_out=fp_size,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.fp_size = fp_size
        self.radius = radius
        self.min_radius = min_radius
        self.rings = rings
        self.isomeric = isomeric
        self.kekulize = kekulize
        self.variant = variant

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
        X = np.array(X, dtype=np.uint32)

        if self.variant in {"bit", "count"}:
            X = np.mod(X, self.fp_size)
            X = np.stack([np.bincount(fp, minlength=self.fp_size) for fp in X])
            if self.variant == "bit":
                X = (X > 0).astype(np.uint8)
            else:
                X = X.astype(np.uint32)

        return csr_array(X) if self.sparse else np.array(X)
