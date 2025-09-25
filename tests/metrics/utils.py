import numpy as np
import pytest
from numpy.testing import assert_equal

from skfp.metrics import extract_pos_proba


def test_extract_pos_proba_single_task():
    n_samples = 10

    predictions = np.random.rand(n_samples, 2)
    predictions = extract_pos_proba(predictions)

    assert_equal(predictions.shape, (n_samples,))


def test_extract_pos_proba_multioutput():
    n_samples = 10
    n_tasks = 5

    predictions = [np.random.rand(n_samples, 2) for _ in range(n_tasks)]
    predictions = extract_pos_proba(predictions)

    assert_equal(predictions.shape, (n_samples, n_tasks))


def test_extract_pos_proba_wrong_dimensions():
    with pytest.raises(
        ValueError, match="Predictions must have 2 or 3 dimensions, got 1"
    ):
        extract_pos_proba([1, 2, 3])

    with pytest.raises(
        ValueError, match="Predictions must have 2 or 3 dimensions, got 4"
    ):
        extract_pos_proba([[[[1, 2, 3]]]])
