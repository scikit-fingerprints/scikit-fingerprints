from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils._param_validation import InvalidParameterError


class BasePreprocessor(ABC, BaseEstimator, TransformerMixin):
    def __sklearn_is_fitted__(self) -> bool:
        return True  # molecule preprocessing transformers don't need fitting

    def fit(self, X, y=None, **fit_params):
        """Unused, kept for Scikit-learn compatibility.

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

    @abstractmethod
    def transform(self, X, copy: bool = False):
        raise NotImplementedError

    def _validate_params(self) -> None:
        # override Scikit-learn validation to make stacktrace nicer
        try:
            super()._validate_params()
        except InvalidParameterError as e:
            raise InvalidParameterError(str(e)) from None
