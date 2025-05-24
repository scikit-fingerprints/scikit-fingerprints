from typing import Optional, Union

import numpy as np
import scipy
from sklearn.utils.validation import check_is_fitted, validate_data

from skfp.bases.base_ad_checker import BaseADChecker


class ConvexHullADChecker(BaseADChecker):
    r"""
    Convex hull method.

    Defines applicability domain based on the convex hull spanned by the training data.
    New molecules should lie inside this space.

    The problem is solved with linear programming formulation [1]_. Problem is reduced to
    question whether a new point can be expressed as a convex linear combination of training
    set points. Formally, for training set of vectors :math:`X={x_1,x_2,...,x_n}` and query
    point `q`, we check whether a problem has any solution:

    - variables :math:`\lambda_i` for :math:`i=1,...,n`
    - we only check if feasible solution exists, setting coefficients :math:`c = 0`
      (all-zeros vector of length :math:`n`)
    - convex combination conditions:

      - :math:`q = \lambda_1 x_1 + ... + \lambda_n x_n`
      - :math:`\lambda_1 + ... + \lambda_n = 1`
      - :math:`\lambda_i \geq 0` for all :math:`i=1,...,n`

    - linear programming formulation:

    .. math::

        \min_\lambda \ & c^T \lambda \\
        \mbox{such that} \ & X \lambda = q,\\
        & 1^T \lambda = 1,\\
        & \lambda_i \geq 0 \text{  for all  } i=1,...,n

    Typically, physicochemical properties (continous features) are used as inputs.
    Consider scaling, normalizing, or transforming them before computing AD to lessen
    effects of outliers, e.g. with ``PowerTransformer`` or ``RobustScaler``.

    This method scales very badly with both number of samples and features. It has
    quadratic scaling :math:`O(n^2)` in number of samples, and can be realistically run
    on at most 1000-3000 molecules. Its geometry also breaks down above ~10 features,
    marking everything as outside AD.

    Parameters
    ----------
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
    .. [1] `StackOverflow discussion - "What's an efficient way to find if a point
        lies in the convex hull of a point cloud?"
        <https://stackoverflow.com/a/43564754/9472066>`_

    Examples
    --------
    >>> import numpy as np
    >>> from skfp.applicability_domain import ConvexHullADChecker
    >>> X_train = np.array([[0.0, 1.0], [0.0, 3.0], [3.0, 1.0]])
    >>> X_test = np.array([[1.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    >>> cvx_hull_ad_checker = ConvexHullADChecker()
    >>> cvx_hull_ad_checker
    ConvexHullADChecker()

    >>> cvx_hull_ad_checker.fit(X_train)
    ConvexHullADChecker()

    >>> cvx_hull_ad_checker.predict(X_test)
    array([ True,  True, False])
    """

    def __init__(
        self,
        n_jobs: Optional[int] = None,
        verbose: Union[int, dict] = 0,
    ):
        super().__init__(
            n_jobs=n_jobs,
            verbose=verbose,
        )

    def fit(  # noqa: D102
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,  # noqa: ARG002
    ):
        X = validate_data(self, X=X)
        self.points_ = X
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: D102
        check_is_fitted(self)
        X = validate_data(self, X=X, reset=False)

        # solution source: https://stackoverflow.com/a/43564754/9472066
        n_points = self.points_.shape[0]

        # coefficients, c^T * lambda, all zeros as we only check feasibility
        c = np.zeros(n_points)

        # equality constraints, A * lambda = b
        # transpose, since lambda is (N, 1) and A is by default (N, D)
        # ones form a row vector that ensure lambdas sum is equal to 1
        # we also add 1s to vector b below
        A = np.vstack([self.points_.T, np.ones(n_points)])

        # variables are non-negative by default in SciPy, so we don't need to
        # explicitly set bounds for lambdas
        results = [
            scipy.optimize.linprog(c, A_eq=A, b_eq=np.hstack([x, 1])).success for x in X
        ]
        return np.array(results)

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
