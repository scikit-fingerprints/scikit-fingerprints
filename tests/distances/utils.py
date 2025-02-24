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
