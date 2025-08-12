from numbers import Real

import numpy as np
from sklearn.base import is_classifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils._param_validation import Interval, InvalidParameterError
from sklearn.utils.validation import check_is_fitted, validate_data

from skfp.bases.base_ad_checker import BaseADChecker


class StandardDeviationADChecker(BaseADChecker):
    """
    Standard deviation method.

    Defines applicability domain based on the spread (standard deviation) of predictions
    from individual estimators in an ensemble model [1]_. The method assumes that
    greater consensus among estimators indicates higher prediction reliability,
    which typically occurs in densely sampled regions of the training data.

    This approach supports both regression models (using ``.predict(X)``) and
    binary classifiers (using ``.predict_proba(X)`` and the probability of the
    positive class). The ensemble model must expose the ``estimators_`` attribute.
    If no model is provided, a default :class:`~sklearn.ensemble.RandomForestRegressor` is created and
    trained during :meth:`fit`.

    At prediction time, each sample is passed to all estimators, and the standard
    deviation of their predictions (or predicted probabilities for classifiers)
    is calculated. The sample is considered in-domain if the standard deviation
    is less than or equal to the specified threshold.

    Parameters
    ----------
    model : object, default=None
        Fitted ensemble model with accessible ``estimators_`` attribute and
        either ``.predict(X)`` or ``.predict_proba(X)`` method on each sub-estimator.
        If not provided, a default :class:`~sklearn.ensemble.RandomForestRegressor` will be created.

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
    >>> std_ad_checker.fit()
    >>> std_ad_checker
    StandardDeviationADChecker()

    >>> X_test = np.random.uniform(0, 1, size=(100, 5))
    >>> std_ad_checker.predict(X_test).shape
    (100,)
    """

    _parameter_constraints: dict = {
        **BaseADChecker._parameter_constraints,
        "model": [object, None],
        "threshold": [Interval(Real, 0, None, closed="left")],
    }

    def __init__(
        self,
        model: object | None = None,
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
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
    ):
        self._validate_params()

        if self.model is None:
            X, y = validate_data(self, X, y, ensure_2d=False)
            self.model_ = RandomForestRegressor(random_state=0)
            self.model_.fit(X, y)  # type: ignore[union-attr]
        else:
            self.model_ = self.model

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: D102
        X = validate_data(self, X=X, reset=False)
        check_is_fitted(self.model_, "estimators_")

        preds = self._predict_all_estimators(X)
        std = np.std(preds, axis=1)

        return (std <= self.threshold).astype(bool)

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
        check_is_fitted(self.model_, "estimators_")
        preds = self._predict_all_estimators(X)
        return np.std(preds, axis=1)

    def _predict_all_estimators(self, X: np.ndarray) -> np.ndarray:
        if is_classifier(self.model_):
            preds = np.array([est.predict_proba(X) for est in self.model_.estimators_])
            preds = preds[:, :, 1]  # shape: (n_estimators, n_samples)
        else:
            preds = np.array([est.predict(X) for est in self.model_.estimators_])

        return preds.T  # shape: (n_samples, n_estimators)

    def _validate_params(self):
        if self.model is not None and is_classifier(self.model):
            check_is_fitted(self.model, "classes_")

            if not hasattr(self.model, "predict_proba"):
                raise InvalidParameterError(
                    f"{self.__class__.__name__} requires classifiers exposing predict_proba."
                )

            if len(getattr(self.model, "classes_", [])) != 2:
                raise InvalidParameterError(
                    f"{self.__class__.__name__} only supports binary classifiers."
                )
