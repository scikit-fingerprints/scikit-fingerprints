import numpy as np
from numpy.testing import assert_allclose

from skfp.metrics import spearman_correlation


def test_spearman_perfect_correlation():
    y_true = list(range(5))
    y_pred = list(range(1, 6))
    corr = spearman_correlation(y_true, y_pred)
    np_corr = spearman_correlation(np.array(y_true), np.array(y_pred))
    assert_allclose(corr, np_corr)
    assert_allclose(corr, 1.0)


def test_spearman_perfect_anticorrelation():
    y_true = list(range(5))
    y_pred = [5, 4, 3, 2, 1]
    corr = spearman_correlation(y_true, y_pred)
    np_corr = spearman_correlation(np.array(y_true), np.array(y_pred))
    assert_allclose(corr, np_corr)
    assert_allclose(corr, -1.0)


def test_spearman_average_correlation():
    y_true = [1, 2, 3, 4]
    y_pred = [2, 1, 4, 3]
    corr = spearman_correlation(y_true, y_pred)
    np_corr = spearman_correlation(np.array(y_true), np.array(y_pred))
    assert_allclose(corr, np_corr)
    assert_allclose(corr, 0.6)


def test_spearman_constant_input():
    y_true = list(range(5))
    y_pred = list(range(5))
    corr = spearman_correlation(y_true, y_pred)
    assert_allclose(corr, 1.0)
    assert np.isnan(spearman_correlation(y_true, y_pred, equal_values_result=np.nan))


def test_spearman_p_value():
    y_true = list(range(5))
    y_pred = list(range(1, 6))
    p_value = spearman_correlation(y_true, y_pred, return_p_value=True)
    assert_allclose(p_value, 0.0)
