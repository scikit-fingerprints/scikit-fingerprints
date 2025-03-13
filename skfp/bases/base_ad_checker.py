from abc import ABC, abstractmethod
from numbers import Integral
from typing import Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin


class BaseADChecker(ABC, BaseEstimator, OutlierMixin):
    """
    Base class for checking applicability domain.

    Applicability Domain (AD) checkers take a list of training molecules and
    model the bounds of their chemical space. For new molecules, they check
    if they lie near the training data, i.e. in the applicability domain
    of the dataset.

    This is very similar to outlier detection, and such methods from e.g.
    scikit-learn can also be applied here. Note that here 0 denotes an outlier
    (outside AD), and 1 means inside AD (inlier). This differs slightly from
    scikit-learn, which uses -1 for outliers, but results in easier filtering.

    This class is not meant to be used directly. If you want to create custom
    applicability domain checkers, inherit from this class and override methods:

    - ``.fit()` - learns the chemical space bounds, e.g. calculates statistics
      saved as object attributes
    - ``.predict()` - checks if molecules are in AD or not, and outputs a vector
      of booleans (True - in AD)
    - ``.score_samples()` - applicability domain score of molecules, lower values
      mean they are more likely outside AD

    Note that score values differ by method, but always should approach zero (at
    least in the limit) if the point is surely outside AD. It can be e.g. number
    of violations or distance to AD.

    Sparse arrays are not supported in most cases, as most methods rely on inherently
    dense physicochemical descriptors.

    Parameters
    ----------
    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`transform_x_y` and
        :meth:`transform` are parallelized over the input molecules. ``None`` means 1
        unless in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors. See scikit-learn documentation on ``n_jobs`` for more details.

    batch_size : int, default=None
        Number of inputs processed in each batch. ``None`` divides input data into
        equal-sized parts, as many as ``n_jobs``.

    verbose : int or dict, default=0
        Controls the verbosity when computing applicability domain statistics.
        If a dictionary is passed, it is treated as kwargs for ``tqdm()``,
        and can be used to control the progress bar.
    """

    # parameters common for all filters
    _parameter_constraints: dict = {
        "n_jobs": [Integral, None],
        "batch_size": [Integral, None],
        "verbose": ["verbose", dict],
    }

    def __init__(
        self,
        n_jobs: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: Union[int, dict] = 0,
    ):
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.verbose = verbose

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Fit applicability domain estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        y : any
            Unused, kept for scikit-learn compatibility.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels (1 inside AD, 0 outside AD) of X according to fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.

        Returns
        -------
        is_inside_applicability_domain : ndarray of shape (n_samples,)
            Returns 1 for molecules inside applicability domain, and 0 for those
            outside (outliers).
        """
        raise NotImplementedError

    @abstractmethod
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the applicability domain score of samples. Higher values indicate
        inliers, well within AD. Lowest possible value is 0, i.e. a definite outlier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Applicability domain scores of samples.

        """
        raise NotImplementedError
