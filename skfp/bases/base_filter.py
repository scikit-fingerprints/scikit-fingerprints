from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from numbers import Integral
from typing import Optional, Union

import numpy as np
from joblib import effective_n_jobs
from rdkit.Chem.rdchem import Mol
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils._param_validation import InvalidParameterError

from skfp.utils import ensure_mols, run_in_parallel, TQDMSettings


class BaseFilter(ABC, BaseEstimator, TransformerMixin):
    """Base class for molecular filters."""

    # parameters common for all filters
    _parameter_constraints: dict = {
        "allow_one_violation": ["boolean"],
        "return_indicators": ["boolean"],
        "n_jobs": [Integral, None],
        "batch_size": [Integral, None],
        "verbose": ["verbose", TQDMSettings],
    }

    def __init__(
        self,
        allow_one_violation: bool = True,
        return_indicators: bool = False,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: Union[int, TQDMSettings] = 0,
    ):
        self.allow_one_violation = allow_one_violation
        self.return_indicators = return_indicators
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.verbose = verbose

    def __sklearn_is_fitted__(self) -> bool:
        return True

    def fit(
        self, X: Sequence[Union[str, Mol]], y: Optional[np.ndarray] = None, **fit_params
    ):
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
        self, X: Sequence[Union[str, Mol]], y: Optional[np.ndarray] = None, **fit_params
    ):
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

    def transform(
        self, X: Sequence[Union[str, Mol]], copy: bool = False
    ) -> Union[list[Union[str, Mol]], np.ndarray]:
        """
        Apply a filter to input molecules. Output depends on ``return_indicators``
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
        filter_ind = self._get_filter_indicators(X, copy)
        if self.return_indicators:
            return filter_ind
        else:
            return [mol for idx, mol in enumerate(X) if filter_ind[idx]]

    def transform_x_y(
        self, X: Sequence[Union[str, Mol]], y: np.ndarray, copy: bool = False
    ) -> tuple[list[Union[str, Mol]], np.ndarray]:
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
        filter_ind = self._get_filter_indicators(X, copy)
        mols = [mol for idx, mol in enumerate(X) if filter_ind[idx]]
        y = y[filter_ind]
        return mols, y

    def _get_filter_indicators(
        self, mols: Sequence[Union[str, Mol]], copy: bool = True
    ) -> np.ndarray:
        self._validate_params()
        mols = deepcopy(mols) if copy else mols
        mols = ensure_mols(mols)

        n_jobs = effective_n_jobs(self.n_jobs)
        if n_jobs == 1:
            filter_indicators = self._filter_mols_batch(mols)
        else:
            filter_indicators = run_in_parallel(
                self._filter_mols_batch,
                data=mols,
                n_jobs=n_jobs,
                batch_size=self.batch_size,
                flatten_results=True,
                verbose=self.verbose,
            )

        return filter_indicators

    def _filter_mols_batch(self, mols: list[Mol]) -> np.ndarray:
        filter_indicators = [self._apply_mol_filter(mol) for mol in mols]
        return np.array(filter_indicators, dtype=bool)

    @abstractmethod
    def _apply_mol_filter(self, mol: Mol) -> bool:
        pass

    def _validate_params(self) -> None:
        # override Scikit-learn validation to make stacktrace nicer
        try:
            super()._validate_params()
        except InvalidParameterError as e:
            raise InvalidParameterError(str(e)) from None
