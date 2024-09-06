from typing import Union

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils._param_validation import (
    Interval,
    RealNotInt,
    StrOptions,
    validate_params,
)


@validate_params(
    {
        "y_true": ["array-like"],
        "y_pred": ["array-like"],
        "constant_target_behavior": [
            StrOptions({"raise"}),
            "nan",
            None,
            Interval(RealNotInt, 0, 1, closed="both"),
        ],
    },
    prefer_skip_nested_validation=True,
)
def auroc_score(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *args,
    constant_target_behavior: Union[str, float] = np.NaN,
    **kwargs,
) -> float:
    """
    Area Under Receiver Operating Characteristic curve (AUROC / ROC AUC).

    Wrapper around scikit-learn `roc_auc_score` function, which adds
    `constant_target_behavior` to control behavior for all-zero `y_true` labels.
    Scikit-learn behavior is to throw an error, since AUROC is undefined there, but
    this can easily happen for cross-validation in imbalanced problems.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_score : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Target scores, i.e. probability of the class with the greater label for each
        output** of the classifier.

    *args, **kwargs
        Any additional parameters for the underlying `roc_auc_score` function.

    constant_target_behavior : "raise", np.NaN, None, or float, default=np.NaN
        Value returned if `y_true` contains only constant values. "raise" is the default
        scikit-learn behavior, which raises an error. Default `np.NaN` (or None) ignores
        the given fold in cross-validation. Alternatively, float value (e.g. 0.5, 1.0) can
        be returned.

    Returns
    -------
    score : float
        AUROC value.

    Examples
    --------
    >>> import numpy as np
    >>> from skfp.metrics import auroc_score
    >>> y_true = np.array([0, 0, 0])
    >>> y_score = np.array([0.5, 0.6, 0.7])
    >>> auroc_score(y_true, y_score)
    nan
    >>> auroc_score(y_true, y_score, constant_target_behavior=0.5)
    0.5
    """

    try:
        return roc_auc_score(y_true, y_score, *args, **kwargs)
    except ValueError as e:
        if (
            "Only one class present in y_true" in str(e)
            and constant_target_behavior != "raise"
        ):
            return constant_target_behavior  # type: ignore
        else:
            raise
