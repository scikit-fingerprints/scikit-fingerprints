import numpy as np
from sklearn.utils.validation import validate_data

from skfp.bases.base_ad_checker import BaseADChecker


class StdEnsembleADChecker(BaseADChecker):
    """
    Standard deviation ensemble method.

    Defines applicability domain based on the spread (standard deviation) of predictions
    from individual estimators in an ensemble model. The method assumes that
    if different estimators give similar predictions for a given molecule,
    the prediction is more likely to be reliable (i.e., in-domain).

    This approach requires an ensemble model exposing the ``estimators_``
    attribute (e.g., RandomForestRegressor), where each
    sub-model must implement a ``predict(X)`` method. At prediction time,
    each sample is passed to all estimators, and the standard deviation of
    their predictions is calculated. The sample is considered in-domain if
    the standard deviation is lower than or equal to the specified threshold.

    Typically, physicochemical properties (continous features) are used as inputs.
    Consider scaling, normalizing, or transforming them before computing AD to lessen
    effects of outliers, e.g. with ``PowerTransformer`` or ``RobustScaler``.

    This method is particularly useful in assessing uncertainty in ensemble
    learning-based QSAR models.

    Parameters
    ----------
    model : object
        Fitted ensemble model with accessible ``estimators_`` attribute and
        ``predict(X)`` method on each sub-estimator.

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

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from skfp.applicability_domain import StdEnsembleADChecker
    >>> X_train = np.random.uniform(0, 1, size=(1000, 5))
    >>> y_train = X_train.sum(axis=1)
    >>> model = RandomForestRegressor(n_estimators=5, random_state=0)
    >>> model.fit(X_train, y_train)
    >>> std_ad_checker = StdEnsembleADChecker(model=model, threshold=0.5)
    >>> std_ad_checker
    StdEnsembleADChecker()

    >>> X_test = np.random.uniform(0, 1, size=(100, 5))
    >>> std_ad_checker.predict(X_test).shape
    (100,)
    """

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
        X: np.ndarray,
        y: np.ndarray | None = None,
    ):
        pass  # unused

    def _predict_all_estimators(self, X: np.ndarray) -> np.ndarray:
        preds = np.array([est.predict(X) for est in self.model.estimators_])
        return preds.T

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: D102
        X = validate_data(self, X=X, reset=False)

        preds = self._predict_all_estimators(X)
        stds = np.std(preds, axis=1)

        return (stds <= self.threshold).astype(bool)

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
