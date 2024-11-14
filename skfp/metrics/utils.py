from typing import Union

import numpy as np
from sklearn.utils._param_validation import validate_params


@validate_params(
    {"predictions": ["array-like"]},
    prefer_skip_nested_validation=True,
)
def extract_pos_proba(predictions: Union[np.ndarray, list[np.ndarray]]) -> np.ndarray:
    """
    Extract positive class probabilities (``y-score``).

    Probabilitic metrics like AUROC or AUPRC use predicted probabilities. This
    function extracts them from ``.predict_proba()`` results.

    Returns `(n_samples,)` shape for single-task, and `(n_samples, n_tasks)` for
    multioutput problems.

    Parameters
    ----------
    predictions : list of NumPy arrays
        Raw predictions of shape ``(n_samples,2)`` (single-task) or
        ``(n_tasks, n_samples, 2)`` (multioutput), with predicted negative
        and positive class probability in the last dimension.

    Returns
    -------
    y_score : NumPy array of shape (n_samples,) or (n_samples, n_tasks
        Predicted positive class probabilities for each task.

    Examples
    --------
    >>> import numpy as np
    >>> from skfp.metrics import extract_pos_proba
    >>> y_pred = np.array([[0.6, 0.1], [0.2, 0.3]])
    >>> extract_pos_proba(y_pred)
    array([0.1, 0.3])
    >>> y_pred = [np.array([[0.6, 0.1], [0.2, 0.3]]), np.array([[0.0, 0.9], [0.7, 0.8]])]
    >>> extract_pos_proba(y_pred)
    array([[0.1, 0.9],
           [0.3, 0.8]])
    """
    # (n_tasks, n_samples, 2) -> (n_tasks, n_samples) -> (n_samples, n_tasks)
    predictions = np.array(predictions)
    if predictions.ndim == 2:
        return predictions[:, 1]
    elif predictions.ndim == 3:
        return np.transpose(predictions[:, :, 1])
    else:
        raise ValueError(
            f"Predictions must have 2 or 3 dimensions, got {predictions.ndim}"
        )
