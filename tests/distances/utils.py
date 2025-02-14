import numpy as np


def assert_similarity_and_distance_values(
    similarity: float, distance: float, comparison: str, value: float
) -> None:
    if comparison == ">":
        assert similarity > value
        assert distance < value
    elif comparison == "<":
        assert similarity < value
        assert distance > value
    elif comparison == "==":
        assert np.isclose(similarity, value)
        assert np.isclose(distance, 1 - value)
