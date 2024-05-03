from collections.abc import Sequence
from copy import deepcopy
from typing import Optional, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from sklearn.utils._param_validation import StrOptions

from skfp.bases import BaseFingerprintTransformer
from skfp.validators import require_mols_with_conf_ids


class USRCATFingerprint(BaseFingerprintTransformer):
    """USRCAT descriptor fingerprint."""

    _parameter_constraints: dict = {
        **BaseFingerprintTransformer._parameter_constraints,
        "errors": [StrOptions({"raise", "NaN", "ignore"})],
    }

    def __init__(
        self,
        errors: str = "raise",
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: int = 0,
    ):
        super().__init__(
            n_features_out=60,
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
        from rdkit.Chem.rdMolDescriptors import GetUSRCAT

        X = require_mols_with_conf_ids(X)

        if self.errors == "raise":
            fps = [GetUSRCAT(mol, confId=mol.GetIntProp("conf_id")) for mol in X]
        else:  # self.errors in {"NaN", "ignore"}
            fps = []
            for mol in X:
                try:
                    fp = GetUSRCAT(mol, confId=mol.GetIntProp("conf_id"))
                except ValueError:
                    fp = np.full(self.n_features_out, np.NaN)
                fps.append(fp)

        return np.array(fps)
