from abc import ABC
from copy import deepcopy
from numbers import Integral
from typing import Optional, Union

from joblib import effective_n_jobs
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils._param_validation import InvalidParameterError

from skfp.utils import run_in_parallel, TQDMSettings


class BasePreprocessor(ABC, BaseEstimator, TransformerMixin):
    """Base class for molecule preprocessing classes."""

    # parameters common for all fingerprints
    _parameter_constraints: dict = {
        "n_jobs": [Integral, None],
        "batch_size": [Integral, None],
        "verbose": ["verbose", TQDMSettings],
    }

    def __init__(
        self,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: Union[int, TQDMSettings] = 0,
    ):
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.verbose = verbose

    def __sklearn_is_fitted__(self) -> bool:
        return True  # molecule preprocessing transformers don't need fitting

    def fit(self, X, y=None, **fit_params):
        """Unused, kept for Scikit-learn compatibility.

        Parameters
        ----------
        X : any
            Unused, kept for Scikit-learn compatibility.

        y : any
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
        The same as ``.transform()`` method, kept for Scikit-learn compatibility.

        Parameters
        ----------
        X : any
            See ``.transform()`` method.

        y : any
            See ``.transform()`` method.

        **fit_params : dict
            Unused, kept for Scikit-learn compatibility.

        Returns
        -------
        X_new : any
            See ``.transform()`` method.
        """
        return self.transform(X)

    def transform(self, X, copy: bool = False):
        self._validate_params()

        if copy:
            X = deepcopy(X)

        n_jobs = effective_n_jobs(self.n_jobs)
        if n_jobs == 1:
            results = self._transform_batch(X)
        else:
            results = run_in_parallel(
                self._transform_batch,
                data=X,
                n_jobs=n_jobs,
                batch_size=self.batch_size,
                flatten_results=True,
                verbose=self.verbose,
            )

        return results

    def _transform_batch(self, X):
        raise NotImplementedError

    def _validate_params(self) -> None:
        # override Scikit-learn validation to make stacktrace nicer
        try:
            super()._validate_params()
        except InvalidParameterError as e:
            raise InvalidParameterError(str(e)) from None
