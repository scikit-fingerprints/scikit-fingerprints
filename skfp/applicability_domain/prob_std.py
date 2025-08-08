from numbers import Real

import numpy as np
from scipy.stats import norm
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import validate_data

from skfp.bases.base_ad_checker import BaseADChecker


class ProbStdADChecker(BaseADChecker):
    """
    Probabilistic standard deviation method (PROB-STD).

    Defines applicability domain based on the probabilistic interpretation of prediction
    uncertainty from individual estimators in an ensemble model [1]_. For each sample,
    the mean and standard deviation of the ensemble predictions are used to construct
    a normal distribution. The score is defined as the probability mass under this
    distribution that lies on the wrong side of the classification threshold (0.5).

    This approach requires a fitted ensemble model exposing the ``estimators_``
    attribute (e.g., RandomForestRegressor or BaggingRegressor), where each
    sub-model implements a ``.predict(X)`` method returning continuous outputs.
    At prediction time, each sample is passed to all estimators, and their predictions
    are used to compute a normal distribution. The sample is considered in-domain if
    the resulting probability of misclassification (PROB-STD) is lower than or equal
    to the specified threshold.

    This method is specifically designed for binary classification with continuous
    predictions around the 0.5 decision threshold, typically from regressors trained
    on binary targets (e.g., 0.0 and 1.0).

    Parameters
    ----------
    model : object
        Fitted ensemble model with accessible ``estimators_`` attribute and
        ``.predict(X)`` method on each sub-estimator.

    threshold : float, default=0.2
        Maximum allowed probability of incorrect class assignment.
        Lower values yield a stricter applicability domain.

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
        Efficiency of different measures for defining the applicability
        domain of classification models. J Cheminform 9, 44 (2017).
        <https://doi.org/10.1186/s13321-017-0230-2>`_

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from skfp.applicability_domain import ProbStdADChecker
    >>> X_train = np.random.uniform(0, 1, size=(1000, 5))
    >>> y_train = (X_train[:, 0] + X_train[:, 1] > 1).astype(float)
    >>> model = RandomForestRegressor(n_estimators=10, random_state=0)
    >>> model.fit(X_train, y_train)
    >>> probstd_ad_checker = ProbStdADChecker(model=model, threshold=0.1)
    >>> probstd_ad_checker
    ProbStdADChecker(model=RandomForestRegressor(...), threshold=0.1)

    >>> X_test = np.random.uniform(0, 1, size=(100, 5))
    >>> probstd_ad_checker.predict(X_test).shape
    (100,)
    """

    _parameter_constraints: dict = {
        **BaseADChecker._parameter_constraints,
        "model": [object],
        "threshold": [Interval(Real, 0, None, closed="left")],
    }

    def __init__(
        self,
        model,
        threshold: float = 0.2,
        n_jobs: int | None = None,
        verbose: int | dict = 0,
    ):
        super().__init__(
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.model = model
        self.threshold = threshold

    def fit(  # noqa: D102
        self,
        X: np.ndarray,  # noqa: ARG002
        y: np.ndarray | None = None,  # noqa: ARG002
    ):
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: D102
        prob_std = self._compute_prob_std(X)
        return prob_std <= self.threshold

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the applicability domain score of samples.
        It is defined as the minimum probabilistic mass under the normal distribution
        that lies on either side of the classification threshold (0.5).
        Lower values indicate higher confidence in class assignment.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Probabilistic scores reflecting the uncertainty of class assignment.
        """
        return self._compute_prob_std(X)

    def _compute_prob_std(self, X: np.ndarray) -> np.ndarray:
        X = validate_data(self, X=X, reset=False)

        preds = np.array([est.predict(X) for est in self.model.estimators_]).T
        y_mean = preds.mean(axis=1)
        y_std = preds.std(axis=1)
        y_std = np.maximum(y_std, 1e-8)

        left_tail = norm.cdf(0.5, loc=y_mean, scale=y_std)
        prob_std = np.minimum(left_tail, 1 - left_tail)
        return prob_std
