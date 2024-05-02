from collections.abc import Sequence
from copy import deepcopy
from typing import Optional, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from sklearn.utils._param_validation import StrOptions

from skfp.bases import BaseFingerprintTransformer
from skfp.validators import require_mols_with_conf_ids


class USRFingerprint(BaseFingerprintTransformer):
    """USR descriptor fingerprint."""

    _parameter_constraints: dict = {
        **BaseFingerprintTransformer._parameter_constraints,
        "errors": [StrOptions({"raise", "NaN", "ignore"})],
    }

    def __init__(
        self,
        errors: str = "raise",
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: int = 0,
    ):
        super().__init__(
            n_features_out=12,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.errors = errors

    def transform(
        self, X: Sequence[Union[str, Mol]], copy: bool = False
    ) -> Union[np.ndarray, csr_array]:
        y = np.empty(len(X))
        X, _ = self.transform_x_y(X, y, copy=copy)
        return X

    def transform_x_y(
        self, X: Sequence[Mol], y: np.ndarray, copy: bool = False
    ) -> tuple[Union[np.ndarray, csr_array], np.ndarray]:
        if copy:
            X = deepcopy(X)
            y = deepcopy(y)

        X = super().transform(X)

        if self.errors == "ignore":
            # errors are marked as NaN rows
            idxs_to_keep = [
                idx for idx, x in enumerate(X) if not np.any(np.isnan(x.data))
            ]
            X = X[idxs_to_keep]
            y = y[idxs_to_keep]

        return X, y

    def _calculate_fingerprint(self, X: Sequence[Mol]) -> Union[np.ndarray, csr_array]:
        from rdkit.Chem.rdMolDescriptors import GetUSR

        X = require_mols_with_conf_ids(X)

        get_usr = lambda mol: GetUSR(mol, confId=mol.GetIntProp("conf_id"))

        if self.errors == "raise":
            fps = [get_usr(mol) for mol in X]
        else:  # self.errors in {"NaN", "ignore"}
            fps = []
            for mol in X:
                try:
                    fp = get_usr(mol)
                except ValueError:
                    fp = np.full(self.n_features_out, np.NaN)
                fps.append(fp)

        return csr_array(fps) if self.sparse else np.array(fps)
