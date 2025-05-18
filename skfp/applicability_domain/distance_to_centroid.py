from numbers import Real
from typing import Callable, Optional, Union

import numpy as np
from scipy.sparse import csr_array
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from sklearn.metrics._dist_metrics import parse_version, sp_base_version
from sklearn.neighbors._base import SCIPY_METRICS as SCIPY_METRIC_NAMES
from sklearn.utils._param_validation import Interval, InvalidParameterError, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data

from skfp.bases.base_ad_checker import BaseADChecker
from skfp.distances import (
    _BULK_METRIC_NAMES as SKFP_BULK_METRIC_NAMES,
)
from skfp.distances import (
    _BULK_METRICS as SKFP_BULK_METRICS,
)
from skfp.distances import (
    _METRIC_NAMES as SKFP_METRIC_NAMES,
)
from skfp.distances import (
    _METRICS as SKFP_METRICS,
)

SCIPY_METRIC_NAMES = set(SCIPY_METRIC_NAMES) | {"euclidean"}
SCIPY_METRICS = {
    "braycurtis": distance.braycurtis,
    "canberra": distance.canberra,
    "chebyshev": distance.chebyshev,
    "correlation": distance.correlation,
    "cosine": distance.cosine,
    "dice": distance.dice,
    "euclidean": distance.euclidean,
    "hamming": distance.hamming,
    "jaccard": distance.jaccard,
    "mahalanobis": distance.mahalanobis,
    "minkowski": distance.minkowski,
    "rogerstanimoto": distance.rogerstanimoto,
    "russellrao": distance.russellrao,
    "seuclidean": distance.seuclidean,
    "sokalsneath": distance.sokalsneath,
    "sqeuclidean": distance.sqeuclidean,
    "yule": distance.yule,
}
if sp_base_version < parse_version("1.17"):
    # Deprecated in SciPy 1.15 and removed in SciPy 1.17
    SCIPY_METRICS["sokalmichener"] = distance.sokalmichener
if sp_base_version < parse_version("1.11"):
    # Deprecated in SciPy 1.9 and removed in SciPy 1.11
    SCIPY_METRICS["kulsinski"] = distance.kulsinski
if sp_base_version < parse_version("1.9"):
    # Deprecated in SciPy 1.0 and removed in SciPy 1.9
    SCIPY_METRICS["matching"] = distance.matching


class DistanceToCentroidADChecker(BaseADChecker):
    r"""
    Distance to centroid method.

    TODO description
    Typically, physicochemical properties (continous features) are used as inputs.
    Consider scaling, normalizing, or transforming them before computing AD to lessen
    effects of outliers, e.g. with ``PowerTransformer`` or ``RobustScaler``.

    TODO scaling

    Parameters
    ----------
    threshold: float or "auto", default="auto"
        Maximal distance allowed for applicability domain. New points with larger
        leverage, i.e. distance to training set, are assumed to lie outside AD.
        ``"auto"`` calculates the distribution of data-centroid distances from the
        training data and uses its 99th percentile.

    metric: str or callable, default="euclidean"
        Metric to use for distance computation. Default is Euclidean distance.
        You can use any scikit-fingerprints distance here, but using bulk variants
        will be faster.

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
    .. [1] `Supratik Kar, Kunal Roy & Jerzy Leszczynski
        "Applicability Domain: A Step Toward Confident Predictions and Decidability for QSAR Modeling"
        In: Nicolotti, O. (eds) Computational Toxicology. Methods in Molecular Biology, vol 1800. Humana Press, New York, NY
        <https://doi.org/10.1007/978-1-4939-7899-1_6>`_

    .. [2] `Paola Gramatica
        "Principles of QSAR models validation: internal and external"
        QSAR & Combinatorial Science 26.5 (2007): 694-701
        <https://doi.org/10.1002/qsar.200610151>`_

    .. [3] `Leverage (statistics) on Wikipedia
        <https://en.wikipedia.org/wiki/Leverage_(statistics)>`_

    Examples
    --------
    >>> import numpy as np
    >>> from skfp.applicability_domain import LeverageADChecker
    >>> X_train = np.array([[0.0, 1.0], [0.0, 3.0], [3.0, 1.0]])
    >>> X_test = np.array([[1.0, 1.0], [1.0, 2.0], [20.0, 3.0]])
    >>> leverage_ad_checker = LeverageADChecker()
    >>> leverage_ad_checker
    LeverageADChecker()

    >>> leverage_ad_checker.fit(X_train)
    LeverageADChecker()

    >>> leverage_ad_checker.predict(X_test)
    array([ True,  True, False])
    """

    _parameter_constraints: dict = {
        **BaseADChecker._parameter_constraints,
        "threshold": [Interval(Real, 0, None, closed="neither"), StrOptions({"auto"})],
        "distance": [
            callable,
            StrOptions(SCIPY_METRIC_NAMES | SKFP_METRIC_NAMES | SKFP_BULK_METRIC_NAMES),
        ],
    }

    def __init__(
        self,
        threshold: Union[float, str] = "auto",
        distance: Union[str, Callable] = "euclidean",
        n_jobs: Optional[int] = None,
        verbose: Union[int, dict] = 0,
    ):
        super().__init__(
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.threshold = threshold
        self.distance = distance

    def _validate_params(self) -> None:
        super()._validate_params()
        if (
            isinstance(self.distance, str)
            and self.distance
            not in SCIPY_METRIC_NAMES | SKFP_METRIC_NAMES | SKFP_BULK_METRIC_NAMES
        ):
            raise InvalidParameterError(
                f"The distance parameter as a string must be one of SciPy metrics "
                f"or scikit-fingerprints metrics. "
                f"SciPy metric names: {SCIPY_METRIC_NAMES}. "
                f"scikit-fingerprints metric names: {SKFP_METRIC_NAMES | SKFP_BULK_METRIC_NAMES}. "
                f"Got: {self.distance}"
            )

    def fit(  # noqa: D102
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,  # noqa: ARG002
    ):
        X = validate_data(self, X=X)

        self.centroid_ = np.mean(X, axis=0).reshape((1, -1))

        if callable(self.distance):
            self.dist_ = lambda arr: cdist(arr, self.centroid_, metric=self.distance)
        elif self.distance in SCIPY_METRIC_NAMES:
            distance = SCIPY_METRICS[self.distance]
            self.dist_ = lambda arr: cdist(arr, self.centroid_, metric=distance)
        elif self.distance in SKFP_METRIC_NAMES:
            distance = SKFP_METRICS[self.distance]
            self.dist_ = lambda arr: cdist(arr, self.centroid_, metric=distance)
        else:
            self.dist_ = SKFP_BULK_METRICS[self.distance]

        if self.threshold == "auto":
            centroid_dists = self.dist_(X)
            self.threshold_ = np.percentile(centroid_dists, q=99)
        else:
            self.threshold_ = self.threshold

        return self

    def predict(self, X: Union[np.ndarray, csr_array]) -> np.ndarray:  # noqa: D102
        centroid_dists = self.score_samples(X)
        return centroid_dists <= self.threshold_

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the applicability domain score of samples. It is equal to the
        distance of each sample to the training data centroid. Note that here lower
        score indicates sample more firmly inside AD.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Applicability domain scores of samples.
        """
        check_is_fitted(self)
        X = validate_data(self, X=X, reset=False)

        centroid_dists = self.dist_(X).ravel()
        return centroid_dists
