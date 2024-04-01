from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
import scipy.sparse
from joblib import delayed, effective_n_jobs
from rdkit.Chem.rdchem import Mol
from rdkit.DataStructs import IntSparseIntVect, LongSparseIntVect, SparseBitVect
from scipy.sparse import csr_array, dok_array
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
)
from sklearn.utils.parallel import Parallel

from skfp.utils import ProgressParallel

"""
If you get MaybeEncodingError, first check any worker functions for exceptions!
That error isn't very informative, but gets thrown in Joblib multiprocessing.
"""

"""
Note that you need to do create RDKit objects *inside* the function that runs
in parallel, i.e. _calculate_fingerprint(), not in the constructor or outside in
general.
This is because Joblib needs to pickle data sent to workers, and RDKit objects
cannot be pickled, throwing TypeError: cannot pickle 'Boost.Python.function' object
"""


class FingerprintTransformer(
    ABC, TransformerMixin, BaseEstimator, ClassNamePrefixFeaturesOutMixin
):
    """Base class for fingerprint transformers."""

    def __init__(
        self,
        n_features_out: int,
        count: bool = False,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        verbose: int = 0,
        random_state: int = 0,
    ):
        self.count = count
        self.sparse = sparse
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

        # this, combined with ClassNamePrefixFeaturesOutMixin, automatically handles
        # set_output() API
        self._n_features_out = n_features_out

    @property
    def n_features_out(self) -> int:
        # publicly expose the number of output features
        # it has underscore at the beginning only due to Scikit-learn convention
        return self._n_features_out

    def __sklearn_is_fitted__(self) -> bool:
        # fingerprint transformers don't require fitting
        return True

    def fit(self, X, y=None, **fit_params):
        return self

    def fit_transform(self, X, y=None, **fit_params):
        return self.transform(X)

    def transform(self, X: Sequence[Union[str, Mol]]) -> Union[np.ndarray, csr_array]:
        """
        :param X: np.array or DataFrame of rdkit.Mol objects
        :return: np.array or sparse array of calculated fingerprints for each molecule
        """
        n_jobs = effective_n_jobs(self.n_jobs)

        if n_jobs == 1:
            return self._calculate_fingerprint(X)
        else:
            batch_size = max(len(X) // n_jobs, 1)

            args = (X[i : i + batch_size] for i in range(0, len(X), batch_size))

            if self.verbose > 0:
                total = min(n_jobs, len(X))
                parallel = ProgressParallel(n_jobs=n_jobs, total=total)
            else:
                parallel = Parallel(n_jobs=n_jobs)

            results = parallel(
                delayed(self._calculate_fingerprint)(X_sub) for X_sub in args
            )

            return scipy.sparse.vstack(results) if self.sparse else np.vstack(results)

    @abstractmethod
    def _calculate_fingerprint(
        self, X: Sequence[Union[str, Mol]]
    ) -> Union[np.ndarray, csr_array]:
        """
        Calculate fingerprints for a given input batch.

        :param X: subset of original X data
        :return: array containing calculated fingerprints for each molecule
        """
        pass

    @staticmethod
    def _hash_fingerprint_bits(
        X: list[Union[IntSparseIntVect, LongSparseIntVect, SparseBitVect]],
        fp_size: int,
        count: bool,
        sparse: bool,
    ) -> Union[np.ndarray, csr_array]:
        if not all(
            isinstance(x, (IntSparseIntVect, LongSparseIntVect, SparseBitVect))
            for x in X
        ):
            raise ValueError(
                "Fingerprint hashing requires instances of one of: "
                "IntSparseIntVect, LongSparseIntVect, SparseBitVect"
            )

        shape = (len(X), fp_size)
        arr = dok_array(shape, dtype=int) if sparse else np.zeros(shape, dtype=int)

        if isinstance(X[0], (IntSparseIntVect, LongSparseIntVect)):
            for idx, x in enumerate(X):
                for fp_bit, val in x.GetNonzeroElements().items():
                    arr[idx, fp_bit % fp_size] += val
        else:
            # SparseBitVect
            for idx, x in enumerate(X):
                for fp_bit in x.GetOnBits():
                    arr[idx, fp_bit % fp_size] += 1

        arr = arr.tocsr() if sparse else arr
        arr = (arr > 0) if not count else arr

        return arr
