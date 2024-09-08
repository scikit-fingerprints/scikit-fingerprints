import inspect
import pickle
from functools import partial
from typing import Type

import numpy as np
from sklearn import clone
from sklearn.utils._testing import create_memmap_backed_data, set_random_state
from sklearn.utils.estimator_checks import (
    check_dont_overwrite_parameters,
    check_estimators_fit_returns_self,
    check_estimators_overwrite_params,
    check_get_params_invariance,
    check_no_attributes_set_in_init,
    check_parameters_default_constructible,
    check_set_params,
)

import skfp.fingerprints
import skfp.preprocessing
from skfp.bases import BaseFilter
from skfp.bases.base_fp_transformer import BaseFingerprintTransformer

"""
Note that many check functions here needed to be copied from Scikit-learn and
slightly modified for compatibility with our input X being a list of SMILES or
RDKit molecules.
"""


# unfortunately, there is no way to pass data to Scikit-learn tests other than
# global variables
n_samples = 10
X = []
y = None


def test_basic_sklearn_checks_for_fingerprints(mols_conformers_list):
    global X, y
    X = mols_conformers_list[:n_samples]
    y = np.arange(n_samples) % 2

    for name, obj in inspect.getmembers(skfp.fingerprints):
        if not inspect.isclass(obj):
            continue

        # USR and USRCAT don't work for molecules with 3 or fewer atoms, so we use
        # NaNs there
        if "USR" in name:
            fp = obj(errors="NaN")
        else:
            fp = obj()

        run_checks(name, fp)


def test_basic_sklearn_checks_for_preprocessors(
    smallest_smiles_list, smallest_mols_list
):
    global X, y
    y = np.arange(n_samples) % 2

    for name, obj in inspect.getmembers(skfp.preprocessing):
        if not inspect.isclass(obj):
            continue

        if "MolFromSmiles" in name:
            X = smallest_smiles_list[:n_samples]
        else:
            X = smallest_mols_list[:n_samples]

        fp = obj()
        run_checks(name, fp)


def run_checks(fp_name: str, fp: BaseFingerprintTransformer):
    checks = [
        check_no_attributes_set_in_init,
        check_fit_score_takes_y,
        check_estimators_fit_returns_self,
        check_estimators_overwrite_params,
        check_estimators_pickle,
        check_transformers_unfitted_stateless,
        check_parameters_default_constructible,
        check_get_params_invariance,
        check_set_params,
        check_dont_overwrite_parameters,
    ]
    for check in checks:
        try:
            check = partial(check, fp_name)
            check(fp)
        except Exception:
            print(f"\n{fp_name} failed check {check.func.__name__}")
            raise


def check_fit_score_takes_y(name: str, estimator_orig: BaseFingerprintTransformer):
    """
    Check that all estimators accept an optional y in fit and score,
    so they can be used in pipelines.
    """
    estimator = clone(estimator_orig)
    set_random_state(estimator)

    funcs = ["fit", "fit_transform"]
    for func_name in funcs:
        func = getattr(estimator, func_name, None)
        if func is not None:
            func(X, y)
            args = [p.name for p in inspect.signature(func).parameters.values()]
            if args[0] == "self":
                # available_if makes methods into functions
                # with an explicit "self", so need to shift arguments
                args = args[1:]
            assert args[1] in ["y", "Y"], (
                "Expected y or Y as second argument for method "
                "%s of %s. Got arguments: %r."
                % (func_name, type(estimator).__name__, args)
            )


def check_estimators_pickle(
    name: str, estimator_orig: BaseFingerprintTransformer, readonly_memmap: bool = False
):
    """
    Test that we can pickle all estimators.
    """
    check_methods = ["predict", "transform", "decision_function", "predict_proba"]

    # nondeterministic, classes, those that cannot be compares with NumPy etc.
    omit_results_check = ["GETAWAYFingerprint", "ConformerGenerator"]

    estimator = clone(estimator_orig)

    set_random_state(estimator)
    estimator.fit(X, y)

    if readonly_memmap:
        unpickled_estimator = create_memmap_backed_data(estimator)
    else:
        # No need to touch the file system in that case.
        pickled_estimator = pickle.dumps(estimator)
        unpickled_estimator = pickle.loads(pickled_estimator)

    result = dict()
    for method in check_methods:
        if hasattr(estimator, method):
            result[method] = getattr(estimator, method)(X)

    for method in result:
        unpickled_result = getattr(unpickled_estimator, method)(X)

        if isinstance(result[method], list):
            assert len(result[method]) == len(unpickled_result)
        else:
            assert result[method].shape == unpickled_result.shape
            if name not in omit_results_check:
                assert np.allclose(
                    result[method], unpickled_result, atol=1e-1, equal_nan=True
                )


def check_transformers_unfitted_stateless(
    name: str, transformer: BaseFingerprintTransformer
):
    """
    Check that using transform without prior fitting
    doesn't raise a NotFittedError for stateless transformers.
    """
    transformer = clone(transformer)
    X_trans = transformer.transform(X)

    if issubclass(transformer.__class__, BaseFilter):
        assert len(X_trans) <= len(X)
    else:
        assert len(X) == len(X_trans)
