import numpy as np


def assert_similarity_values(similarity: float, comparison: str, value: float) -> None:
    if comparison == ">":
        assert similarity > value
    elif comparison == "<":
        assert similarity < value
    elif comparison == "==":
        assert np.isclose(similarity, value, atol=1e-3)


def assert_distance_values(distance: float, comparison: str, value: float) -> None:
    if comparison == ">":
        assert distance < value
    elif comparison == "<":
        assert distance > value
    elif comparison == "==":
        assert np.isclose(distance, value, atol=1e-3)


def assert_matrix_similarity_values(
    similarity_matrix: np.ndarray, comparison: str, value_matrix: np.ndarray
) -> None:
    assert similarity_matrix.shape == value_matrix.shape

    if comparison == ">":
        assert np.all(similarity_matrix > value_matrix)
    elif comparison == "<":
        assert np.all(similarity_matrix < value_matrix)
    elif comparison == "==":
        assert np.allclose(similarity_matrix, value_matrix, atol=1e-3)


def assert_matrix_distance_values(
    distance_matrix: np.ndarray, comparison: str, value_matrix: np.ndarray
) -> None:
    assert distance_matrix.shape == value_matrix.shape

    if comparison == ">":
        assert np.all(distance_matrix > value_matrix)
    elif comparison == "<":
        assert np.all(distance_matrix < value_matrix)
    elif comparison == "==":
        assert np.allclose(distance_matrix, value_matrix, atol=1e-3)
