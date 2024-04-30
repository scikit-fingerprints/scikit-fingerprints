from collections.abc import Sequence
from copy import deepcopy
from typing import Optional, Union

import numpy as np
from joblib import effective_n_jobs
from rdkit.Chem import Mol
from scipy import sparse
from scipy.sparse import csr_array
from sklearn.utils._param_validation import InvalidParameterError

from skfp.bases import BaseFingerprintTransformer
from skfp.parallel import run_in_parallel
from skfp.validators import require_mols_with_conf_ids


class USRDescriptor(BaseFingerprintTransformer):
    """USR descriptor fingerprint."""

    def __init__(
        self,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: int = 0,
        use_usr_cat: bool = False,
        errors: str = "NaN",
    ):
        if use_usr_cat:
            n_features_out = 60
        else:
            n_features_out = 12
        super().__init__(
            n_features_out=n_features_out,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.use_usr_cat = use_usr_cat
        self.errors = errors

    def _compute_usr(self, X: Sequence[Mol]) -> Union[np.ndarray, csr_array]:
        if self.use_usr_cat:
            from rdkit.Chem.rdMolDescriptors import GetUSRCAT

            fp_function = GetUSRCAT
        else:
            from rdkit.Chem.rdMolDescriptors import GetUSR

            fp_function = GetUSR

        X = require_mols_with_conf_ids(X)

        if self.errors == "raise":
            transformed_mols = [fp_function(mol) for mol in X]
        else:
            transformed_mols = []
            for mol in X:
                try:
                    mol_fp = fp_function(mol)
                except ValueError:
                    mol_fp = np.empty((self.n_features_out,))
                    mol_fp.fill(np.nan)
                finally:
                    transformed_mols.append(mol_fp)

        if self.sparse:
            return csr_array(transformed_mols)
        else:
            return np.array(transformed_mols)

    def _remove_molecule_nans(
        self, X: Union[np.ndarray, csr_array], y: np.ndarray
    ) -> tuple[Union[np.ndarray, csr_array], np.ndarray]:
        idxs_to_keep = []
        for idx, x in enumerate(X):
            if not np.isnan(x[0]):
                idxs_to_keep.append(idx)
        y = y[idxs_to_keep]
        X = X[idxs_to_keep]
        return X, y

    def _calculate_fingerprint(self, X: Sequence[Mol]) -> Union[np.ndarray, csr_array]:
        X = self._compute_usr(X)
        if self.errors == "ignore":
            X, _ = self._remove_molecule_nans(X, np.zeros(len(X)))

        return X

    def transform_x_y(
        self, X: Sequence[Mol], y: np.ndarray, copy: bool = False
    ) -> tuple[Union[np.ndarray, csr_array], np.ndarray]:
        if self.errors == "raise":
            InvalidParameterError(
                "To transform molecules with provided labels,"
                'use "ignore" or "NaN" value for the `errors` parameter'
            )

        if copy:
            X = deepcopy(X)

        n_jobs = effective_n_jobs(self.n_jobs)
        if n_jobs == 1:
            result_X = self._compute_usr(X)
        else:
            result_X = run_in_parallel(
                self._compute_usr,
                data=X,
                n_jobs=n_jobs,
                batch_size=self.batch_size,
                verbose=self.verbose,
            )
            result_X = sparse.vstack(result_X) if self.sparse else np.vstack(result_X)

        if self.errors == "ignore":
            result_X, y = self._remove_molecule_nans(result_X, y)

        return result_X, y
