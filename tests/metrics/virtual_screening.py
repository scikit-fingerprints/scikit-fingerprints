import numpy as np

from skfp.metrics import bedroc_score, enrichment_factor, rie_score


def test_enrichment_factor():
    y_test = [0] * 90 + [1] * 10
    y_score = [0] * 95 + [1] * 5
    enrichment_factor(y_test, y_score)  # smoke test, should just work


def test_zero_enrichment_factor():
    y_test = [0] * 90 + [1] * 10
    y_score = [1] * 90 + [0] * 10
    ef = enrichment_factor(y_test, y_score)
    assert np.isclose(ef, 0, atol=1e-4)


def test_max_enrichment_factor():
    y_test = [0] * 90 + [1] * 10
    y_score = [0] * 90 + [1] * 10

    # if X >= n/N, max value is N/n
    max_ef = len(y_test) / sum(y_test)

    ef = enrichment_factor(y_test, y_score)
    assert np.isclose(ef, max_ef)


def test_rie_score():
    y_test = [0] * 90 + [1] * 10
    y_score = [0] * 95 + [1] * 5
    rie_score(y_test, y_score)  # smoke test, should just work


def test_zero_rie_score():
    y_test = [0] * 90 + [1] * 10
    y_score = [1] * 90 + [0] * 10
    rie = rie_score(y_test, y_score)
    assert np.isclose(rie, 0, atol=1e-4)


def test_max_rie_score():
    y_test = [0] * 90 + [1] * 10
    y_score = [0] * 90 + [1] * 10

    N = len(y_test)
    n = sum(y_test)
    alpha = 20
    max_rie = (N / n) * (1 - np.e ** (-alpha * n / N)) / (1 - np.e ** (-alpha))

    rie = rie_score(y_test, y_score, alpha)
    assert np.isclose(rie, max_rie)


def test_bedroc_score():
    y_test = [0] * 90 + [1] * 10
    y_score = [0] * 95 + [1] * 5
    bedroc_score(y_test, y_score)  # smoke test, should just work


def test_zero_bedroc_score():
    y_test = [0] * 90 + [1] * 10
    y_score = [1] * 90 + [0] * 10
    bedroc = bedroc_score(y_test, y_score)
    assert np.isclose(bedroc, 0, atol=1e-4)


def test_max_bedroc_score():
    y_test = [0] * 90 + [1] * 10
    y_score = [0] * 90 + [1] * 10
    bedroc = bedroc_score(y_test, y_score)
    assert np.isclose(bedroc, 1)
