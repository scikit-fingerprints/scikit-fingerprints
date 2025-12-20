from numbers import Integral

import numpy as np
from scipy.sparse import csr_array
from sklearn.decomposition import PCA
from sklearn.utils._param_validation import Interval, RealNotInt, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data

from skfp.bases.base_ad_checker import BaseADChecker


class PCABoundingBoxADChecker(BaseADChecker):
    r"""
    PCA bounding box method.

    Defines the applicability domain using coordinate axes found by Principal Component
    Analysis (PCA) [1]_. AD is a hyperrectangle, with thresholds defined as minimal and
    maximal values from the training set on PCA axes.

    Typically, physicochemical properties (continuous features) are used as inputs.
    Consider scaling, normalizing, or transforming them before computing AD to lessen
    effects of outliers, e.g. with ``PowerTransformer`` or ``RobustScaler``.

    This method scales very well with both the number of samples and features, but doesn't
    work well for highly sparse input features (e.g. many molecular fingerprints),
    since PCA centers the data.

    Parameters
    ----------
    n_components: int, float or "mle", default=None
        Number of PCA dimensions (components) used, passed to underlying scikit-learn
        PCA instance. `scikit-learn PCA documentation <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`_
        for more information.

    whiten: bool, default=False
        When True (False by default) the `components_` vectors are multiplied by the
        square root of `n_samples` and then divided by the singular values to ensure
        uncorrelated outputs with unit component-wise variances.

        Whitening will remove some information from the transformed signal (the relative
        variance scales of the components), but this can sometimes improve the robustness
        of applicability domain.

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
    .. [1] `Jaworska J, Nikolova-Jeliazkova N, Aldenberg T.
        "QSAR Applicability Domain Estimation by Projection of the Training Set
        in Descriptor Space: A Review"
        Alternatives to Laboratory Animals. 2005;33(5):445-459
        <https://journals.sagepub.com/doi/10.1177/026119290503300508>`_

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
    }

    def __init__(
        self,
        n_components: int | float | str | None = None,
        whiten: bool = False,
        n_jobs: int | None = None,
        verbose: int | dict = 0,
    ):
        super().__init__(
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.n_components = n_components
        self.whiten = whiten

    def fit(  # noqa: D102
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,  # noqa: ARG002
    ):
        X = validate_data(self, X=X)

        self.pca_ = PCA(
            n_components=self.n_components,
            whiten=self.whiten,
            random_state=0,
        )
        self.pca_.fit(X)

        X_transformed = self.pca_.transform(X)
        self.min_vals_ = X_transformed.min(axis=0)
        self.max_vals_ = X_transformed.max(axis=0)

        return self

    def predict(self, X: np.ndarray | csr_array) -> np.ndarray:  # noqa: D102
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
