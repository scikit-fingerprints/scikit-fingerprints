from collections.abc import Sequence
from numbers import Integral
from typing import Optional, Union

import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from sklearn.base import BaseEstimator, _fit_context, clone
from sklearn.model_selection._search import BaseSearchCV, ParameterSampler
from sklearn.utils._param_validation import Interval, InvalidParameterError

from skfp.bases import BaseFingerprintTransformer


class FingerprintEstimatorRandomizedSearch(BaseEstimator):
    _parameter_constraints: dict = {
        "fingerprint": [BaseFingerprintTransformer],
        "estimator_cv": [BaseSearchCV],
        "fp_param_distributions": [dict, list],
        "n_iter": [Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        fingerprint: BaseFingerprintTransformer,
        estimator_cv: BaseSearchCV,
        fp_param_distributions: Union[dict, list[dict]],
        n_iter: int = 10,
        random_state: Optional[int] = 0,
    ):
        self.fingerprint = fingerprint
        self.estimator_cv = estimator_cv
        self.fp_param_distributions = fp_param_distributions
        self.n_iter = n_iter
        self.random_state = random_state

    def _validate_params(self):
        super()._validate_params()
        if not self.fp_param_distributions:
            raise InvalidParameterError("fp_param_distributions cannot not be empty")

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X: Sequence[Union[str, Mol]], y=None, **params):
        self.cv_results_: list[dict] = []
        self.best_fp_: BaseFingerprintTransformer = None  # type: ignore
        self.best_fp_params_: dict = None  # type: ignore
        self.best_score_ = -1  # in scikit-learn, higher score is always better
        self.best_estimator_cv_: BaseSearchCV = None  # type: ignore

        for fp_params in ParameterSampler(
            self.fp_param_distributions, self.n_iter, random_state=self.random_state
        ):
            fp: BaseFingerprintTransformer = clone(self.fingerprint)
            fp.set_params(**fp_params)

            X_fp = fp.fit_transform(X, y, **params)

            curr_cv = clone(self.estimator_cv)
            curr_cv.fit(X_fp, y)
            curr_score = curr_cv.best_score_
            estimator_params = curr_cv.best_params_

            result = {**fp_params, **estimator_params, "score": curr_score}
            self.cv_results_.append(result)

            if curr_score > self.best_score_:
                self.best_fp_ = fp
                self.best_fp_params_ = fp_params
                self.best_score_ = curr_score
                self.best_estimator_cv_ = curr_cv

        return self

    def predict(self, X: Sequence[Union[str, Mol]]) -> np.ndarray:
        X_fp = self.best_fp_.transform(X)
        return self.best_estimator_cv_.predict(X_fp)

    def predict_proba(self, X: Sequence[Union[str, Mol]]) -> np.ndarray:
        X_fp = self.best_fp_.transform(X)
        return self.best_estimator_cv_.predict_proba(X_fp)

    def transform(self, X: Sequence[Union[str, Mol]]) -> Union[np.ndarray, csr_array]:
        return self.best_fp_.transform(X)
