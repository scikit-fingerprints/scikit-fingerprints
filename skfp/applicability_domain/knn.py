from collections.abc import Callable
from numbers import Integral

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils._param_validation import Interval, InvalidParameterError, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data

from skfp.bases.base_ad_checker import BaseADChecker
from skfp.distances import (
    tanimoto_binary_distance,
    tanimoto_count_distance,
)

METRIC_FUNCTIONS = {
    "tanimoto_binary": tanimoto_binary_distance,
    "tanimoto_count": tanimoto_count_distance,
}


class KNNADChecker(BaseADChecker):
    r"""
    k-Nearest Neighbor applicability domain checker.

    This method determines whether a query molecule falls within the applicability
    domain by comparing its distance to k nearest neighbors [1]_ [2]_ in the training set,
    using a threshold derived from the training data.

    The applicability domain is defined as either:
     - the mean distance to k nearest neighbors,
     - the max distance among the k nearest neighbors,
     - the min [3]_ distance among the k nearest neighbors (effectively kNN with k of 1)

    for each training sample. A threshold is then set at the
    95th percentile of these aggregated distances. Query molecules with an aggregated
    distance to their k nearest neighbors below this threshold are considered within
    the applicability domain.

    This implementation supports binary and count Tanimoto similarity metrics.

    Parameters
    ----------
    k : int
        Number of nearest neighbors to consider for distance calculations.
        Must be smaller than the number of training samples.

    metric: {"tanimoto_binary", "tanimoto_count"}, default="tanimoto_binary"
        Distance metric to use.

    agg: {"mean", "max", "min"}, default="mean"
        Aggregation method for distances to k nearest neigbors:
            - "mean": use the mean distance to k neighbors,
            - "max": use the maximum distance among k neighbors,
            - "min": use the distance to the closest neigbor.

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`transform_x_y` and
        :meth:`transform` are parallelized over the input molecules. ``None`` means 1
        unless in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors. See scikit-learn documentation on ``n_jobs`` for more details.

    verbose : int or dict, default=0
        Controls the verbosity when filtering molecules.
        If a dictionary is passed, it is treated as kwargs for ``tqdm()``,
        and can be used to control the progress bar.

    References
    ----------

    .. [1] `Klingspohn, W., Mathea, M., ter Laak, A. et al.
        "Efficiency of different measures for defining the applicability domain of
        classification models."
        J Cheminform 9, 44 (2017)
        <https://doi.org/10.1186/s13321-017-0230-2>`_

    .. [2] `Harmeling, S., Dornhege G., Tax D., Meinecke F., Müller KR.
        "From outliers to prototypes: Ordering data."
        Neurocomputing, 69, 13, pages 1608-1618, (2006)
        <https://doi.org/10.1016/j.neucom.2005.05.015>`_

    .. [3] `Kar S., Roy K., Leszczynski J.
        "Applicability Domain: A Step Toward Confident Predictions and Decidability for QSAR Modeling"
        Methods Mol Biol, 1800, pages 141-169, (2018)
        <https://doi.org/10.1007/978-1-4939-7899-1_6>`_

    Examples
    --------
    >>> from skfp.applicability_domain import KNNADChecker
    >>> import numpy as np
    >>> X_train_binary = np.array([
    ...     [1, 1, 1],
    ...     [0, 1, 1],
    ...     [0, 0, 1]
    ... ])
    >>> X_test_binary = 1 - X_train_binary
    >>> knn_ad_checker_binary = KNNADChecker(k=2, metric="tanimoto_binary", agg="mean")
    >>> knn_ad_checker_binary
    KNNADChecker()

    >>> knn_ad_checker_binary.fit(X_train_binary)
    KNNADChecker()

    >>> knn_ad_checker_binary.predict(X_test_binary)
    array([False, False, False])

    >>> X_train_count = np.array([
    ...     [1.2, 2.3],
    ...     [3.4, 4.5],
    ...     [5.6, 6.7]
    ... ])
    >>> X_test_count = X_train_count + 10
    >>> knn_ad_checker_count = KNNADChecker(k=2, metric="tanimoto_count", agg="min")
    >>> knn_ad_checker_count
    KNNADChecker()

    >>> knn_ad_checker_count.fit(X_train_count)
    KNNADChecker()

    >>> knn_ad_checker_count.predict(X_test_count)
    array([False, False, False])

    """

    _parameter_constraints: dict = {
        **BaseADChecker._parameter_constraints,
        "k": [Interval(Integral, 1, None, closed="left")],
        "metric": [
            callable,
            StrOptions(set(METRIC_FUNCTIONS.keys())),
        ],
    }

    def __init__(
        self,
        k: int,
        metric: str | Callable = "tanimoto_binary",
        agg: str = "mean",
        n_jobs: int | None = None,
        verbose: int | dict = 0,
    ):
        super().__init__(
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.metric = metric
        self.agg = agg
        self.k = k

    def _validate_params(self) -> None:
        super()._validate_params()
        if isinstance(self.metric, str) and self.metric not in METRIC_FUNCTIONS:
            raise InvalidParameterError(
                f"The metric parameter must be one of Tanimoto variants. "
                f"Allowed Tanimoto metrics: {list(METRIC_FUNCTIONS.keys())}. "
                f"Got: {self.metric}"
            )
        if isinstance(self.agg, str) and self.agg not in ["mean", "max", "min"]:
            raise InvalidParameterError("Unknown aggregration method.")

    def fit(  # noqa: D102
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,  # noqa: ARG002
    ):
        X = validate_data(self, X=X)
        if self.k >= X.shape[0]:
            raise ValueError(
                f"k ({self.k}) must be smaller than the number of training samples ({X.shape[0]})"
            )

        k_used = 1 if self.agg == "min" else self.k

        if callable(self.metric):
            metric_func = self.metric
        elif isinstance(self.metric, str) and self.metric in METRIC_FUNCTIONS:
            metric_func = METRIC_FUNCTIONS[self.metric]
        else:
            raise InvalidParameterError(
                f"Unknown metric: {self.metric}. Must be a callable or one of {list(METRIC_FUNCTIONS.keys())}"
            )

        self.knn_ = NearestNeighbors(
            n_neighbors=k_used, metric=metric_func, n_jobs=self.n_jobs
        )
        self.knn_.fit(X)

        dists, _ = self.knn_.kneighbors(X)

        if self.agg == "mean":
            agg_dists = np.mean(dists, axis=1)
        elif self.agg == "max":
            agg_dists = np.max(dists, axis=1)
        elif self.agg == "min":
            agg_dists = np.min(dists, axis=1)

        self.threshold_ = np.percentile(agg_dists, 95)

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: D102
        check_is_fitted(self)
        X = validate_data(self, X=X, reset=False)

        k_used = 1 if self.agg == "min" else self.k
        dists, _ = self.knn_.kneighbors(X, n_neighbors=k_used)
        if self.agg == "mean":
            agg_dists = np.mean(dists, axis=1)
        elif self.agg == "max":
            agg_dists = np.max(dists, axis=1)
        elif self.agg == "min":
            agg_dists = np.min(dists, axis=1)

        return agg_dists <= self.threshold_

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the applicability domain score of samples. It is simply a 0/1
        decision equal to ``.predict()``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Applicability domain scores of samples.
        """
        return self.predict(X)
