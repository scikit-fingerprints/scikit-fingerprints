import numpy as np
import pytest
from numpy.testing import assert_allclose

from skfp.metrics import bedroc_score, enrichment_factor, rie_score


def test_enrichment_factor():
    y_test = [0] * 90 + [1] * 10
    y_score = [0] * 95 + [1] * 5
    enrichment_factor(y_test, y_score)  # smoke test, should just work


def test_zero_enrichment_factor():
    y_test = [0] * 90 + [1] * 10
    y_score = [1] * 90 + [0] * 10
    ef = enrichment_factor(y_test, y_score)
    assert_allclose(ef, 0, atol=1e-4)


def test_max_enrichment_factor():
    y_test = [0] * 90 + [1] * 10
    y_score = [0] * 90 + [1] * 10

    # if X >= n/N, max value is N/n
    max_ef = len(y_test) / sum(y_test)

    ef = enrichment_factor(y_test, y_score)
    assert_allclose(ef, max_ef)


def test_rie_score():
    y_test = [0] * 90 + [1] * 10
    y_score = [0] * 95 + [1] * 5
    rie_score(y_test, y_score)  # smoke test, should just work


def test_zero_rie_score():
    y_test = [0] * 90 + [1] * 10
    y_score = [1] * 90 + [0] * 10
    rie = rie_score(y_test, y_score)
    assert_allclose(rie, 0, atol=1e-4)


def test_max_rie_score():
    y_test = [0] * 90 + [1] * 10
    y_score = [0] * 90 + [1] * 10

    N = len(y_test)
    n = sum(y_test)
    alpha = 20
    max_rie = (N / n) * (1 - np.e ** (-alpha * n / N)) / (1 - np.e ** (-alpha))

    rie = rie_score(y_test, y_score, alpha)
    assert_allclose(rie, max_rie)


def test_bedroc_score():
    y_test = [0] * 90 + [1] * 10
    y_score = [0] * 95 + [1] * 5
    bedroc_score(y_test, y_score)  # smoke test, should just work


def test_zero_bedroc_score():
    y_test = [0] * 90 + [1] * 10
    y_score = [1] * 90 + [0] * 10
    bedroc = bedroc_score(y_test, y_score)
    assert_allclose(bedroc, 0, atol=1e-4)


def test_max_bedroc_score():
    y_test = [0] * 90 + [1] * 10
    y_score = [0] * 90 + [1] * 10
    bedroc = bedroc_score(y_test, y_score)
    assert_allclose(bedroc, 1)


@pytest.mark.parametrize("metric_func", [enrichment_factor, rie_score, bedroc_score])
def test_multiclass_inputs(metric_func):
    y_test = [0] * 80 + [1] * 10 + [2] * 10
    y_score = [0] * 80 + [1] * 10 + [2] * 10
    with pytest.raises(ValueError, match="defined for binary y_true"):
        metric_func(y_test, y_score)
