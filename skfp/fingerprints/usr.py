from collections.abc import Sequence
from copy import deepcopy
from typing import Optional, Union

import numpy as np
import scipy.sparse
from joblib import effective_n_jobs
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from sklearn.utils._param_validation import StrOptions

from skfp.bases import BaseFingerprintTransformer
from skfp.parallel import run_in_parallel
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

        # we have no easy way to pass multiple arguments into parallel function,
        # so we pass list of tuples instead and unpack them later
        data = list(zip(X, y))

        n_jobs = effective_n_jobs(self.n_jobs)
        if n_jobs == 1:
            X, y = self._calculate_fingerprint(data)  # type: ignore
        else:
            results = run_in_parallel(
                self._calculate_fingerprint,
                data=data,
                n_jobs=n_jobs,
                batch_size=self.batch_size,
                verbose=self.verbose,
            )
            X, y = list(zip(*results))  # type: ignore
            X = scipy.sparse.vstack(X) if self.sparse else np.vstack(X)
            y = np.concatenate(y)

        return X, y

    def _calculate_fingerprint(
        self, X: Sequence[tuple[Mol, np.ndarray]]  # type: ignore
    ) -> tuple[Union[np.ndarray, csr_array], np.ndarray]:
        from rdkit.Chem.rdMolDescriptors import GetUSR

        X, y = list(zip(*X))
        X = list(X)
        y = np.array(y)

        X = require_mols_with_conf_ids(X)

        get_usr = lambda mol: GetUSR(mol, confId=mol.GetIntProp("conf_id"))

        if self.errors == "raise":
            fps = [get_usr(mol) for mol in X]
        elif self.errors == "NaN":
            fps = []
            for mol in X:
                try:
                    fp = get_usr(mol)
                except ValueError:
                    fp = np.full(self.n_features_out, np.NaN)
                fps.append(fp)
        else:  # self.errors == "ignore"
            fps = []
            idxs_to_keep = []
            for idx, mol in enumerate(X):
                try:
                    fps.append(get_usr(mol))
                    idxs_to_keep.append(idx)
                except ValueError:
                    pass
            y = y[idxs_to_keep] if idxs_to_keep else np.empty(0)

        if self.sparse:
            return csr_array(fps, dtype=np.float32), y
        else:
            return np.array(fps, dtype=np.float32), y
