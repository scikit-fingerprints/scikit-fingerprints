from collections.abc import Sequence
from time import time
from typing import Union

import joblib.logger
import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from sklearn.base import BaseEstimator, _fit_context, clone
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection._search import BaseSearchCV
from sklearn.utils._param_validation import InvalidParameterError

from skfp.bases import BaseFingerprintTransformer


class FingerprintEstimatorGridSearch(BaseEstimator):
    """
    Exhaustive search over specified hyperparameter values for a pipeline of a molecular
    fingerprint and scikit-learn estimator.

    This approach is useful for pipelines which first compute fingerprints and then
    operate on the resulting matrices, and when both fingerprint and estimator
    hyperparameters are optimized. Regular scikit-learn combination of ``Pipeline`` and
    ``GridSearchCV`` would recompute the fingerprint for each set of hyperparameter values.

    Here, we instead perform a nested loop:

    1. Loop over all possible combinations of fingerprint hyperparameter values
    2. Compute fingerprint
    3. Optimize estimator hyperparameters

    This way, computed fingerprint representations are efficiently used for many sets of
    estimator hyperparameters. This is useful when tuning classifier or fingerprint and
    classifier. When only fingerprint is tuned, combination of ``Pipeline`` and
    ``GridSearchCV`` is enough. The difference is particularly significant for more
    computationally heavy fingerprints and large grids for estimators.

    Note that much of the behavior is controlled via passed ``estimator_cv`` object, e.g.
    the ``scoring`` metric used to select the best pipeline. In particular, if
    ``GridSearchCV`` is used, then the result is equivalent to using grid search on all
    hyperparameters, but faster. However, any other strategy can be used for the
    estimator, e.g. ``RandomizedSearchCV``.

    Parameters
    ----------
    fingerprint : fingerprint object
        Instance of any fingerprint class. To maximize performance, consider setting
        `n_jobs` larger than 1, since parallelization is not performed here when going
        through fingerprint hyperparameter grid.

    fp_param_grid : dict or list[dict]
        Dictionary with names of fingerprint hyperparameters as keys and lists of
        hyperparameter settings to try as values, or a list of such dictionaries, in which
        case the grids spanned by each dictionary in the list are explored. This enables
        searching over any sequence of hyperparameter settings.

    estimator_cv : object
        Inner cross-validation object for tuning estimator, e.g. ``GridSearchCV``.
        Should be an instantiated object, not a class.

    greater_is_better : bool, default=True
        Whether higher values of scoring metric in ``estimator_cv`` are better or not.
        ``False`` should be used for error (loss) functions, typically used in regression.

    cache_best_fp_array : bool, default=False
        Whether to cache the array of values from the best fingerprint in ``best_fp_array_``
        parameter. Note that this can result in high memory usage.

    verbose : int or dict, default=0
        Controls the verbosity when computing fingerprints.

        - >0 : size of parameter grid, parameter candidate for each fold
        - >1 : the computation time and score for each candidate

        If a dictionary is passed, it is treated as kwargs for ``tqdm()``,
        and can be used to control the progress bar.

    Attributes
    ----------
    cv_results_ : list[dict]
        List of dictionaries, where each one represents the set of hyperparameters
        (names and values) and ``"score"`` key with the cross-validated performance of the
        pipeline with those hyperparameters.

    best_fp_ : fingerprint object
        Fingerprint that was chosen by the search, i.e. fingerprint which gave the highest
        score (or smallest loss if specified) on the left out data. Use with
        ``best_estimator_cv_`` to obtain the best found pipeline.

    best_fp_params_ : dict
        Fingerprint hyperparameter values that gave the best results on the hold out data.

    best_fp_array_ : np.ndarray
        Fingerprint values for ``best_fp_``. If ``cache_best_fp_array`` is False, this will not
        be used and will be None instead.

    best_score_ : float
        Mean cross-validated score of the best fingerprint and estimator.

    best_estimator_cv_ : CV object
        Inner cross-validation object that gave the best results on the hold out data. Use
        with ``best_fp_`` to obtain the best found pipeline.

    See Also
    --------
    :class:`FingerprintEstimatorRandomizedSearch` : Related fingerprint, but uses
        randomized search for fingerprint hyperparameters.

    Examples
    --------
    >>> from skfp.datasets.moleculenet import load_bace
    >>> from skfp.fingerprints import ECFPFingerprint
    >>> from skfp.model_selection import FingerprintEstimatorGridSearch
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.model_selection import GridSearchCV
    >>> smiles, labels = load_bace()
    >>> fp = ECFPFingerprint(n_jobs=-1)
    >>> fp_params = {"radius": [2, 3]}
    >>> clf = RandomForestClassifier(n_jobs=-1)
    >>> clf_params = {"min_samples_split": [2, 3, 4]}
    >>> clf_cv = GridSearchCV(clf, clf_params)
    >>> fp_cv = FingerprintEstimatorGridSearch(fp, fp_params, clf_cv)
    >>> fp_cv.fit(smiles, labels)  # doctest: +SKIP
    FingerprintEstimatorGridSearch(estimator_cv=GridSearchCV(estimator=RandomForestClassifier(n_jobs=-1),
                                                             param_grid={'min_samples_split': [2,
                                                                                               3,
                                                                                               4]}),
                                   fingerprint=ECFPFingerprint(n_jobs=-1),
                                   fp_param_grid={'radius': [2, 3]})
    >>> fp_cv.best_fp_params_  # doctest: +SKIP
    {'radius': 2}
    """

    _parameter_constraints: dict = {
        "fingerprint": [BaseFingerprintTransformer],
        "fp_param_grid": [dict, list],
        "estimator_cv": [BaseSearchCV],
        "greater_is_better": ["boolean"],
        "cache_best_fp_array": ["boolean"],
        "verbose": ["verbose", dict],
    }

    def __init__(
        self,
        fingerprint: BaseFingerprintTransformer,
        fp_param_grid: Union[dict, list[dict]],
        estimator_cv: BaseSearchCV,
        greater_is_better: bool = True,
        cache_best_fp_array: bool = False,
        verbose: Union[int, dict] = 0,
    ):
        self.fingerprint = fingerprint
        self.fp_param_grid = fp_param_grid
        self.estimator_cv = estimator_cv
        self.greater_is_better = greater_is_better
        self.cache_best_fp_array = cache_best_fp_array
        self.verbose = verbose

    def _validate_params(self):
        super()._validate_params()
        if not self.fp_param_grid:
            raise InvalidParameterError("fp_param_grid cannot not be empty")

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X: Sequence[Union[str, Mol]], y=None, **params):
        self.cv_results_: list[dict] = []
        self.best_fp_: BaseFingerprintTransformer = None  # type: ignore
        self.best_fp_params_: dict = None  # type: ignore
        self.best_fp_array_: np.ndarray = None  # type: ignore
        self.best_score_ = None
        self.best_estimator_cv_: BaseSearchCV = None  # type: ignore

        param_grid = ParameterGrid(self.fp_param_grid)
        param_grid_size = len(param_grid)
        if self.verbose:
            print(f"Fitting {param_grid_size} candidate hyperparameter sets.")

        for idx, fp_params in enumerate(param_grid):
            if self._print_messages():
                self._print_start_msg(
                    curr_idx=idx + 1,
                    grid_size=param_grid_size,
                    curr_params=fp_params,
                )
            start_time = time()

            fp: BaseFingerprintTransformer = clone(self.fingerprint)
            fp.set_params(**clone(fp_params, safe=False))

            X_fp = fp.fit_transform(X, y, **params)

            curr_cv = clone(self.estimator_cv)
            curr_cv.fit(X_fp, y, **params)
            curr_score = curr_cv.best_score_
            estimator_params = curr_cv.best_params_

            result = {**fp_params, **estimator_params, "score": curr_score}
            self.cv_results_.append(result)

            if (
                (self.best_score_ is None)
                or (self.greater_is_better and curr_score > self.best_score_)
                or (not self.greater_is_better and curr_score < self.best_score_)
            ):
                self.best_fp_ = fp
                self.best_fp_params_ = fp_params
                self.best_score_ = curr_score
                self.best_estimator_cv_ = curr_cv
                if self.cache_best_fp_array:
                    self.best_fp_array_ = X_fp

            end_time = time()
            if self._print_messages():
                self._print_end_msg(
                    curr_idx=idx + 1,
                    grid_size=param_grid_size,
                    curr_params=fp_params,
                    start_time=start_time,
                    end_time=end_time,
                    curr_score=curr_score,
                )

        return self

    def predict(self, X: Sequence[Union[str, Mol]]) -> np.ndarray:
        X_fp = self.best_fp_.transform(X)
        return self.best_estimator_cv_.predict(X_fp)

    def predict_proba(self, X: Sequence[Union[str, Mol]]) -> np.ndarray:
        X_fp = self.best_fp_.transform(X)
        return self.best_estimator_cv_.predict_proba(X_fp)

    def transform(self, X: Sequence[Union[str, Mol]]) -> Union[np.ndarray, csr_array]:
        return self.best_fp_.transform(X)

    def _print_messages(self) -> bool:
        if isinstance(self.verbose, int):
            return self.verbose > 0
        elif isinstance(self.verbose, dict):
            return len(self.verbose) > 0

    def _print_start_msg(
        self, curr_idx: int, grid_size: int, curr_params: dict
    ) -> None:
        progress_msg = f"{curr_idx}/{grid_size}"
        params_msg = ", ".join(
            f"{name}={curr_params[name]}" for name in sorted(curr_params)
        )
        start_msg = f"[{progress_msg}] START {params_msg}"
        print(f"{start_msg}{(80 - len(start_msg)) * '.'}")

    def _print_end_msg(
        self,
        curr_idx: int,
        grid_size: int,
        curr_params: dict,
        start_time: float,
        end_time: float,
        curr_score: float,
    ) -> None:
        progress_msg = f"{curr_idx}/{grid_size}"
        params_msg = ", ".join(
            f"{name}={curr_params[name]}" for name in sorted(curr_params)
        )
        score_msg = f"score={curr_score:.3f}"
        total_time = joblib.logger.short_format_time(end_time - start_time)
        time_msg = f"total time={total_time}"
        end_msg = f"[{progress_msg}] END {params_msg}; {score_msg}; {time_msg}"
        print(f"{end_msg}{(80 - len(end_msg)) * '.'}")
