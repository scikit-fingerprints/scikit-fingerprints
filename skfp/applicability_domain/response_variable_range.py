from numbers import Real

import numpy as np
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import check_is_fitted, validate_data

from skfp.bases.base_ad_checker import BaseADChecker


class ResponseVariableRangeADChecker(BaseADChecker):
    """
    Response variable range method.

    Defines applicability domain based on the range of response values observed
    in the training data [1]_. New predictions are considered inside the applicability
    domain if they lie within the min-max range of training targets.

    Typically, this method is used after model prediction, and checks whether
    predicted values lie within the known domain of the response variable.

    Note that this method does not consider molecular structure or descriptors.
    It operates purely in the output (target) space.

    This method scales extremely well with the number of samples,
    as it only operates in the 1D target space.

    Parameters
    ----------
    threshold : float, default=None
        Maximum allowed distance from the training response mean.
        If float, defines a symmetric interval around the mean:
        `[mean - threshold, mean + threshold]`, and predictions outside
        this range are considered outside the applicability domain.
        If ``None`` (default), the method uses the full min–max range
        of training targets as bounds.

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
    >>> from skfp.applicability_domain import ResponseVariableRangeADChecker
    >>> y_train = np.array([0.5, 1.2, 1.5, 0.9])
    >>> y_pred = np.array([1.0, 1.6, 0.4])
    >>> response_range_ad_checker = ResponseVariableRangeADChecker()
    >>> response_range_ad_checker
    ResponseVariableRangeADChecker()

    >>> response_range_ad_checker.fit(y_train)
    ResponseVariableRangeADChecker()

    >>> response_range_ad_checker.predict(y_pred)
    array([ True, False, False])
    """

    _parameter_constraints: dict = {
        **BaseADChecker._parameter_constraints,
        "threshold": [None, Interval(Real, 0, None, closed="left")],
    }

    def __init__(
        self,
        threshold: float | None = None,
        n_jobs: int | None = None,
        verbose: int | dict = 0,
    ):
        super().__init__(
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.threshold = threshold

    def fit(  # type: ignore[override]  # noqa: D102
        self,
        y: np.ndarray,
        X: np.ndarray | None = None,  # noqa: ARG002
    ):
        y = validate_data(self, X=y, ensure_2d=False)

        self.mean_y_ = np.mean(y)

        if self.threshold is None:
            self.lower_bound_ = np.min(y)
            self.upper_bound_ = np.max(y)
        else:
            self.lower_bound_ = self.mean_y_ - self.threshold
            self.upper_bound_ = self.mean_y_ + self.threshold

        return self

    def predict(self, y: np.ndarray) -> np.ndarray:  # noqa: D102
        check_is_fitted(self)
        y = validate_data(self, X=y, ensure_2d=False, reset=False)
        return (self.lower_bound_ <= y) & (self.upper_bound_ >= y)

    def score_samples(self, y: np.ndarray) -> np.ndarray:
        """
        Calculate the applicability domain score of samples. It is defined as
        the absolute distance of each predicted value from the training response mean.
        Lower scores indicate that a prediction is closer to the center of the training
        response distribution.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Predicted response values.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Applicability domain scores of samples.
        """
        y = validate_data(self, X=y, ensure_2d=False, reset=False)
        return np.abs(y - self.mean_y_)
