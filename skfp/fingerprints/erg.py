from numbers import Integral, Real
from typing import Optional, Sequence, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from sklearn.utils import Interval
from sklearn.utils._param_validation import InvalidParameterError

from skfp.validators import ensure_mols

from .base import FingerprintTransformer


class ERGFingerprint(FingerprintTransformer):
    """Extended Reduced Graph Fingerprint (ERG) transformer."""

    _parameter_constraints: dict = {
        **FingerprintTransformer._parameter_constraints,
        "fuzz_increment": [Interval(Real, 0.0, None, closed="left")],
        "min_path": [Interval(Integral, 1, None, closed="left")],
        "max_path": [Interval(Integral, 1, None, closed="left")],
    }

    def __init__(
        self,
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
        self.fuzz_increment = fuzz_increment
        self.min_path = min_path
        self.max_path = max_path

    def _validate_params(self) -> None:
        super()._validate_params()
        if self.max_path < self.min_path:
            raise InvalidParameterError(
                f"The max_path parameter of {self.__class__.__name__} must be"
                f"greater or equal to min_path, got: "
                f"min_path={self.min_path}, max_path={self.max_path}"
            )

    def _calculate_fingerprint(
        self, X: Sequence[Union[str, Mol]]
    ) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem.rdReducedGraphs import GetErGFingerprint

        X = ensure_mols(X)

        X = [
            GetErGFingerprint(
                x,
                fuzzIncrement=self.fuzz_increment,
                minPath=self.min_path,
                maxPath=self.max_path,
            )
            for x in X
        ]

        return csr_array(X) if self.sparse else np.array(X)
