import numpy as np
from scipy.sparse import csr_array
from scipy.stats import f as f_dist
from sklearn.utils._param_validation import Interval, RealNotInt
from sklearn.utils.validation import check_is_fitted, validate_data

from skfp.bases.base_ad_checker import BaseADChecker


class HotellingT2TestADChecker(BaseADChecker):
    r"""
    Hotelling's T^2 test method.

    Applicability domain is defined by the Hotelling's T^2 statistical test, which
    is a multidimensional generalization of Student's t-test [1]_. It measures the
    Mahalanobis distance of a new sample from the mean of the training data, scaled
    by the covariance structure of the training data.

    Typically, physicochemical properties (continuous features) are used as inputs.
    Consider scaling, normalizing, or transforming them before computing AD to lessen
    effects of outliers, e.g. with ``PowerTransformer`` or ``RobustScaler``. In case
    of Hotelling's T^2 test, using PCA beforehand to obtain orthogonal features is
    particularly beneficial.

    This method scales relatively well with number of samples, but not the number of
    features, as it requires computing the pseudoinverse of the covariance matrix,
    which scales as :math:`O(d^3)`.

    Parameters
    ----------
    alpha: float, default=0.1
        Statistical test significance level, in range ``[0, 1]``. Lower values are
        more conservative and correspond to a smaller applicability domain.

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
    >>> from skfp.applicability_domain import HotellingT2TestADChecker
    >>> X_train = np.array([[0.0, 1.0], [0.0, 3.0], [3.0, 1.0]])
    >>> X_test = np.array([[1.0, 1.0], [1.0, 2.0], [20.0, 3.0]])
    >>> hotelling_t2_test_ad_checker = HotellingT2TestADChecker()
    >>> hotelling_t2_test_ad_checker
    HotellingT2TestADChecker()

    >>> hotelling_t2_test_ad_checker.fit(X_train)
    HotellingT2TestADChecker()

    >>> hotelling_t2_test_ad_checker.predict(X_test)
    array([ True,  True, False])
    """

    _parameter_constraints: dict = {
        **BaseADChecker._parameter_constraints,
        "alpha": [Interval(RealNotInt, 0, 1, closed="neither")],
    }

    def __init__(
        self,
        alpha: float = 0.1,
        n_jobs: int | None = None,
        verbose: int | dict = 0,
    ):
        super().__init__(
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.alpha = alpha

    def fit(  # noqa: D102
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ):
        X = validate_data(self, X=X)

        n, p = X.shape
        self.X_mean_ = np.mean(X, axis=0)

        # get inverse of covariance matrix for Mahalanobis distance
        cov = np.cov(X, rowvar=False)  # rowvar=False: rows are samples
        self.inv_cov_ = np.linalg.pinv(cov)  # pseudoinverse for numerical stability

        # threshold based on critical value from the F-distribution
        f_val = f_dist.ppf(self.alpha, p, n - p)
        self.threshold_ = p * (n - 1) * (n + 1) / (n * (n - p)) * f_val

        return self

    def predict(self, X: np.ndarray | csr_array) -> np.ndarray:  # noqa: D102
        t2_values = self.score_samples(X)
        passed = t2_values <= self.threshold_
        return passed

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the applicability domain score of samples, i.e. the
        T^2 statistic value for samples. Note that here lower score indicates
        sample more firmly inside AD.

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

        X_centered = X - self.X_mean_
        t2_values = np.array([(x @ self.inv_cov_ @ x.T) for x in X_centered])

        return t2_values
