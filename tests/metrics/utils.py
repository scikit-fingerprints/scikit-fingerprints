import numpy as np

from skfp.metrics import extract_pos_proba


def test_extract_pos_proba_single_task():
    n_samples = 10

    predictions = np.random.rand(n_samples, 2)
    predictions = extract_pos_proba(predictions)

    assert predictions.shape == (n_samples,)


def test_extract_pos_proba_multioutput():
    n_samples = 10
    n_tasks = 5

    predictions = [np.random.rand(n_samples, 2) for _ in range(n_tasks)]
    predictions = extract_pos_proba(predictions)

    assert predictions.shape == (n_samples, n_tasks)
