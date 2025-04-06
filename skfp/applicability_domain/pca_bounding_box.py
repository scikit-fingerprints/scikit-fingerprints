from numbers import Integral
from typing import Optional, Union

import numpy as np
from scipy.sparse import csr_array
from sklearn.decomposition import PCA
from sklearn.utils._param_validation import Interval, RealNotInt, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data

from skfp.bases.base_ad_checker import BaseADChecker


class PCABoundingBoxADChecker(BaseADChecker):
    r"""
    PCA bounding box method.

    TODO description

    Typically, physicochemical properties (continous features) are used as inputs.
    Consider scaling, normalizing, or transforming them before computing AD to lessen
    effects of outliers, e.g. with ``PowerTransformer`` or ``RobustScaler``.

    TODO scaling

    Parameters
    ----------
    threshold: float or "auto", default="auto"
        Maximal leverage allowed for applicability domain. New points with larger
        leverage, i.e. distance to training set, are assumed to lie outside AD.

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
    .. [1]

    Examples
    --------
    >>> import numpy as np
    >>> from skfp.applicability_domain import PCABoundingBoxADChecker
    >>> X_train = np.array([[0.0, 1.0], [0.0, 3.0], [3.0, 1.0]])
    >>> X_test = np.array([[1.0, 1.0], [1.0, 2.0], [20.0, 3.0]])
    >>> pca_bb_ad_checker = PCABoundingBoxADChecker()
    >>> pca_bb_ad_checker
    PCABoundingBoxADChecker()

    >>> pca_bb_ad_checker.fit(X_train)
    PCABoundingBoxADChecker()

    >>> pca_bb_ad_checker.predict(X_test)
    array([ True,  True, False])
    """

    _parameter_constraints: dict = {
        **BaseADChecker._parameter_constraints,
        "n_components": [
            Interval(Integral, 0, None, closed="left"),
            Interval(RealNotInt, 0, 1, closed="neither"),
            StrOptions({"mle"}),
            None,
        ],
        "whiten": ["boolean"],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        n_components: Union[int, float, str, None] = None,
        whiten: bool = False,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
        verbose: Union[int, dict] = 0,
    ):
        super().__init__(
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.n_components = n_components
        self.whiten = whiten
        self.random_state = random_state

    def fit(  # noqa: D102
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,  # noqa: ARG002
    ):
        X = validate_data(self, X=X)

        self.pca_ = PCA(
            n_components=self.n_components,
            whiten=self.whiten,
            random_state=self.random_state,
        )
        self.pca_.fit(X)

        X_transformed = self.pca_.transform(X)
        self.min_vals_ = X_transformed.min(axis=0)
        self.max_vals_ = X_transformed.max(axis=0)

        return self

    def predict(self, X: Union[np.ndarray, csr_array]) -> np.ndarray:  # noqa: D102
        check_is_fitted(self)
        X = validate_data(self, X=X, reset=False)

        X_transformed = self.pca_.transform(X)

        passed = (X_transformed >= self.min_vals_) & (X_transformed <= self.max_vals_)
        passed = np.all(passed, axis=1)

        return passed

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
