from numbers import Real

import numpy as np
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import validate_data

from skfp.bases.base_ad_checker import BaseADChecker


class StandardDeviationADChecker(BaseADChecker):
    """
    Standard deviation method.

    Defines applicability domain based on the spread (standard deviation) of predictions
    from individual estimators in an ensemble model [1]_. The method assumes that
    greater consensus among estimators indicates higher prediction reliability,
    which typically occurs in densely sampled regions of the training data.

    This approach requires an ensemble model exposing the ``estimators_``
    attribute (e.g., RandomForestRegressor), where each
    sub-model must implement a ``.predict(X)`` method. At prediction time,
    each sample is passed to all estimators, and the standard deviation of
    their predictions is calculated. The sample is considered in-domain if
    the standard deviation is lower than or equal to the specified threshold.

    Parameters
    ----------
    model : object
        Fitted ensemble model with accessible ``estimators_`` attribute and
        ``.predict(X)`` method on each sub-estimator.

    threshold : float, default=1.0
        Maximum allowed standard deviation of predictions. Samples with
        higher spread will be considered outside the applicability domain.

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
    .. [1] `Kar, S., Roy, K., Leszczynski, J.
        "Applicability Domain: A Step Toward Confident Predictions and Decidability
        for QSAR Modeling."
        Nicolotti, O. (eds) Computational Toxicology. Methods in Molecular Biology, vol 1800.
        Humana Press, New York, NY
        <https://doi.org/10.1007/978-1-4939-7899-1_6>`_

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from skfp.applicability_domain import StandardDeviationADChecker
    >>> X_train = np.random.uniform(0, 1, size=(1000, 5))
    >>> y_train = X_train.sum(axis=1)
    >>> model = RandomForestRegressor(n_estimators=5, random_state=0)
    >>> model.fit(X_train, y_train)
    >>> std_ad_checker = StandardDeviationADChecker(model=model, threshold=0.5)
    >>> std_ad_checker
    StandardDeviationADChecker()

    >>> X_test = np.random.uniform(0, 1, size=(100, 5))
    >>> std_ad_checker.predict(X_test).shape
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
        threshold: float = 1.0,
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
        X = validate_data(self, X=X, reset=False)

        preds = self._predict_all_estimators(X)
        stds = np.std(preds, axis=1)

        return (stds <= self.threshold).astype(bool)

    def _predict_all_estimators(self, X: np.ndarray) -> np.ndarray:
        preds = np.array([est.predict(X) for est in self.model.estimators_])
        return preds.T

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the applicability domain score of samples.
        It is defined as the standard deviation of ensemble predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data matrix.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
             Standard deviations of predictions.
        """
        X = validate_data(self, X=X, reset=False)
        preds = self._predict_all_estimators(X)
        return np.std(preds, axis=1)
