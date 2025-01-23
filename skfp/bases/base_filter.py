from abc import ABC, abstractmethod
from collections.abc import Sequence
from copy import deepcopy
from numbers import Integral
from typing import Optional, Union

import numpy as np
from joblib import effective_n_jobs
from rdkit.Chem import Mol
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils._param_validation import InvalidParameterError
from tqdm import tqdm

from skfp.utils import ensure_mols, run_in_parallel


class BaseFilter(ABC, BaseEstimator, TransformerMixin):
    """
    Base class for molecular filters.

    Filters take a list of molecules and check which ones fulfill the conditions
    specified. It can be used to create both common types of filters:

    - "pass" filters, where molecules have to fit into a given range of properties,
      e.g. Lipinski Rule of 5
    - "reject" filters, where molecules cannot contain certain properties like
      toxic functional groups, e.g. PAINS filters

    This class is not meant to be used directly. If you want to create custom
    filters, inherit from this class and override the ``._apply_mol_filter()``
    method. It gets a single molecule and outputs a boolean, whether it passes
    the filter or not. Note that for "reject" filters it should return True if
    molecule should be kept, i.e. does not contain any undesirable property.

    Parameters
    ----------
    allow_one_violation : bool, default=False
        Whether to allow violating one of the rules for a molecule. This makes the
        filter less restrictive.

    return_indicators : bool, default=False
        Whether to return a binary vector with indicators which molecules pass the
        filter, instead of list of molecules.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`transform_x_y` and
        :meth:`transform` are parallelized over the input molecules. ``None`` means 1
        unless in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors. See scikit-learn documentation on ``n_jobs`` for more details.

    batch_size : int, default=None
        Number of inputs processed in each batch. ``None`` divides input data into
        equal-sized parts, as many as ``n_jobs``.

    verbose : int or dict, default=0
        Controls the verbosity when filtering molecules.
        If a dictionary is passed, it is treated as kwargs for ``tqdm()``,
        and can be used to control the progress bar.
    """

    # parameters common for all filters
    _parameter_constraints: dict = {
        "allow_one_violation": ["boolean"],
        "return_indicators": ["boolean"],
        "n_jobs": [Integral, None],
        "batch_size": [Integral, None],
        "verbose": ["verbose", dict],
    }

    def __init__(
        self,
        allow_one_violation: bool = False,
        return_indicators: bool = False,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: Union[int, dict] = 0,
    ):
        self.allow_one_violation = allow_one_violation
        self.return_indicators = return_indicators
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.verbose = verbose

    def __sklearn_is_fitted__(self) -> bool:
        """
        Unused, kept for scikit-learn compatibility. This class assumes stateless
        transformers and always returns True.
        """
        return True

    def fit(
        self, X: Sequence[Union[str, Mol]], y: Optional[np.ndarray] = None, **fit_params
    ):
        """Unused, kept for scikit-learn compatibility.

        Parameters
        ----------
        X : any
            Unused, kept for scikit-learn compatibility.

        y : any
            Unused, kept for scikit-learn compatibility.

        **fit_params : dict
            Unused, kept for scikit-learn compatibility.

        Returns
        -------
        self
        """
        self._validate_params()
        return self

    def fit_transform(
        self, X: Sequence[Union[str, Mol]], y: Optional[np.ndarray] = None, **fit_params
    ):
        """
        The same as ``.transform()`` method, kept for scikit-learn compatibility.

        Parameters
        ----------
        X : any
            See ``.transform()`` method.

        y : any
            See ``.transform()`` method.

        **fit_params : dict
            Unused, kept for scikit-learn compatibility.

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
            Sequence containing RDKit ``Mol`` objects.

        copy : bool, default=False
            Copy the input X or not.

        Returns
        -------
        X : list of shape (n_samples_conf_gen,) or array of shape (n_samples,)
            List with filtered molecules, or indicator vector which molecules
            fulfill the filter rules.
        """
        filter_ind = self._get_filter_indicators(X, copy)
        if self.return_indicators:
            return filter_ind
        else:
            return [mol for idx, mol in enumerate(X) if filter_ind[idx]]

    def transform_x_y(
        self, X: Sequence[Union[str, Mol]], y: np.ndarray, copy: bool = False
    ) -> Union[tuple[list[Union[str, Mol]], np.ndarray], tuple[np.ndarray, np.ndarray]]:
        """
        Apply a filter to input molecules. Output depends on ``return_indicators``
        attribute.

        Parameters
        ----------
        X : {sequence, array-like} of shape (n_samples,)
            Sequence containing RDKit ``Mol`` objects.

        y : array-like of shape (n_samples,)
            Array with labels for molecules.

        copy : bool, default=False
            Copy the input X or not.

        Returns
        -------
        X : list of shape (n_samples_conf_gen,) or array of shape (n_samples,)
            List with filtered molecules, or indicator vector which molecules
            fulfill the filter rules.

        y : np.ndarray of shape (n_samples_conf_gen,)
            Array with labels for molecules.
        """
        filter_ind = self._get_filter_indicators(X, copy)
        mols = [mol for idx, mol in enumerate(X) if filter_ind[idx]]
        y = y[filter_ind]
        if self.return_indicators:
            return filter_ind, y
        else:
            return mols, y

    def _get_filter_indicators(
        self, mols: Sequence[Union[str, Mol]], copy: bool
    ) -> np.ndarray:
        self._validate_params()
        mols = deepcopy(mols) if copy else mols
        mols = ensure_mols(mols)

        n_jobs = effective_n_jobs(self.n_jobs)
        if n_jobs == 1:
            if self.verbose:
                filter_indicators = [
                    self._filter_mols_batch([mol]) for mol in tqdm(mols)
                ]
            else:
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
        # override scikit-learn validation to make stacktrace nicer
        try:
            super()._validate_params()
        except InvalidParameterError as e:
            raise InvalidParameterError(str(e)) from None
