from abc import abstractmethod
from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
from rdkit.Chem import Mol
from sklearn.utils._param_validation import InvalidParameterError

from skfp.bases import BasePreprocessor


class BaseFilter(BasePreprocessor):
    def __sklearn_is_fitted__(self) -> bool:
        return True  # molecule preprocessing transformers don't need fitting

    def fit(self, X: Sequence[Mol], y: Optional[np.ndarray] = None, **fit_params):
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

    def fit_transform(
        self, X: Sequence[Mol], y: Optional[np.ndarray] = None, **fit_params
    ):
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
        self, X: Sequence[Mol], copy: bool = False
    ) -> Union[list[Mol], np.ndarray]:
        """
        Apply a filter to input molecules. Output depends on `return_indicators`
        attribute.

        Parameters
        ----------
        X : {sequence, array-like} of shape (n_samples,)
            Sequence containing RDKit Mol objects.

        copy : bool, default=True
            Copy the input X or not.

        Returns
        -------
        X : list of shape (n_samples_conf_gen,) or array of shape (n_samples,)
            List with filtered RDKit Mol objects, or indicator vector which molecules
            fulfill the filter rules.
        """
        filter_indicators = self._filter(X, copy)
        if self.return_indicators:
            return filter_indicators
        else:
            return [mol for idx, mol in enumerate(X) if filter_indicators[idx]]

    @abstractmethod
    def transform_x_y(
        self, X: Sequence[Mol], y: np.ndarray, copy: bool = False
    ) -> tuple[Union[list[Mol], np.ndarray], np.ndarray]:
        """
        Apply a filter to input molecules.

        Parameters
        ----------
        X : {sequence, array-like} of shape (n_samples,)
            Sequence containing RDKit Mol objects.

        y : array-like of shape (n_samples,)
            Array with labels for molecules.

        copy : bool, default=True
            Copy the input X or not.

        Returns
        -------
        X : list of shape (n_samples_conf_gen,) or array of shape (n_samples,)
            List with filtered RDKit Mol objects, or indicator vector which molecules
            fulfill the filter rules.

        y : np.ndarray of shape (n_samples_conf_gen,)
            Array with labels for molecules.
        """
        filter_indicators = self._filter(X, copy)
        mols = [mol for idx, mol in enumerate(X) if filter_indicators[idx]]
        y = y[filter_indicators]
        return mols, y

    @abstractmethod
    def _filter(self, mols: Sequence[Mol], copy: bool = True) -> np.ndarray:
        pass

    def _validate_params(self) -> None:
        # override Scikit-learn validation to make stacktrace nicer
        try:
            super()._validate_params()
        except InvalidParameterError as e:
            raise InvalidParameterError(str(e)) from None
