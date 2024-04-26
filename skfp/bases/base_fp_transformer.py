from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from numbers import Integral
from typing import Optional, Union

import numpy as np
import scipy.sparse
from joblib import effective_n_jobs
from rdkit.Chem.rdchem import Mol
from rdkit.DataStructs import IntSparseIntVect, LongSparseIntVect, SparseBitVect
from scipy.sparse import csr_array, dok_array
from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
)
from sklearn.utils._param_validation import InvalidParameterError

from skfp.parallel import run_in_parallel

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


class BaseFingerprintTransformer(
    ABC, BaseEstimator, TransformerMixin, ClassNamePrefixFeaturesOutMixin
):
    """Base class for fingerprint transformers."""

    # parameters common for all fingerprints
    _parameter_constraints: dict = {
        "count": ["boolean"],
        "sparse": ["boolean"],
        "n_jobs": [Integral, None],
        "batch_size": [Integral, None],
        "verbose": ["verbose"],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        n_features_out: int,
        requires_conformers: bool = False,
        count: bool = False,
        sparse: bool = False,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: int = 0,
        random_state: Optional[int] = 0,
    ):
        self.count = count
        self.sparse = sparse
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.verbose = verbose
        self.random_state = random_state

        # this, combined with ClassNamePrefixFeaturesOutMixin, automatically handles
        # set_output() API
        self._n_features_out = n_features_out
        self.n_features_out = self._n_features_out

        # indicate whether inputs need to be molecules with conformers computed and
        # conf_id integer property set; this allows programmatically checking which
        # fingerprints are 3D-based and require such input
        self.requires_conformers = requires_conformers

    def __sklearn_is_fitted__(self) -> bool:
        return True  # fingerprint transformers don't need fitting

    def set_params(self, **params):
        super().set_params(**params)
        # for fingerprints that have both 2D and 3D versions, as indicated by use_3D
        # attribute, we need to also keep requires_conformers attribute in sync
        if hasattr(self, "use_3D"):
            self.requires_conformers = self.use_3D
        return self

    def fit(self, X, y=None, **fit_params):
        """
        Unused, kept for Scikit-learn compatibility.

        Parameters
        ----------
        X : any
            Unused, kept for Scikit-learn compatibility.

        Y : any
            Unused, kept for Scikit-learn compatibility.

        **fit_params : dict
            Unused, kept for Scikit-learn compatibility.

        Returns
        --------
        self
        """
        self._validate_params()
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """
        The same as `transform` method, kept for Scikit-learn compatibility.

        Parameters
        ----------
        X : any
            See `transform` method.

        y : any
            See `transform` method.

        **fit_params : dict
            Unused, kept for Scikit-learn compatibility.

        Returns
        -------
        X_new : any
            See `transform` method.
        """
        return self.transform(X)

    def transform(
        self, X: Sequence[Union[str, Mol]], copy: bool = False
    ) -> Union[np.ndarray, csr_array]:
        self._validate_params()

        if copy:
            X = deepcopy(X)

        n_jobs = effective_n_jobs(self.n_jobs)
        if n_jobs == 1:
            results = self._calculate_fingerprint(X)
        else:
            results = run_in_parallel(
                self._calculate_fingerprint,
                data=X,
                n_jobs=n_jobs,
                batch_size=self.batch_size,
                verbose=self.verbose,
            )

        if isinstance(results, (np.ndarray, csr_array)):
            return results
        else:
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
        raise NotImplementedError

    def _validate_params(self) -> None:
        # override Scikit-learn validation to make stacktrace nicer
        try:
            super()._validate_params()
        except InvalidParameterError as e:
            raise InvalidParameterError(str(e)) from None

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
        dtype = np.uint32 if count else np.uint8
        arr = dok_array(shape, dtype=dtype) if sparse else np.zeros(shape, dtype=dtype)

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
        arr = (arr > 0).astype(np.uint8) if not count else arr

        return arr
