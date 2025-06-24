import numpy as np
from numpy.typing import NDArray as ndarray

# Author: Jeroen Faas


def data_n_vertices(data: ndarray[float], number_of_vertices: ndarray[int], n: int) -> ndarray[float]:
    """
    Extracts the given data with a corresponding given number of vertices equal to "n".
    :param data: A numpy.ndarray of shape (N,), where entry i equals the data of section profile i. This section
            profile i should correspond to entry i in "number_of_vertices".
    :param number_of_vertices: A numpy.ndarray of shape (N,), where entry i equals the number of vertices of section
            profile i. This section profile i should correspond to entry i in "data".
    :param n: An integer, being the specified number of vertices of interest.
    :return: A numpy.ndarray of shape (M,), containing the data of all section profiles with exactly "n" vertices.
    """
    return data[number_of_vertices == n]


def count_pos_numbers(numbers: ndarray[int], fractions: bool = False) -> ndarray[int]:
    """
    Counts the occurrences of all positive numbers in a list of integers.
    :param numbers: A numpy.ndarray of shape (N,), containing integers.
    :param fractions: A boolean, indicating whether to return counts or densities. False by default.
    :return: A numpy.ndarray of shape (M - m + 1,), containing the counts if "fractions" is False, or the fractions if
            "fractions" is True, of all positive numbers from m up to and including M. Here, m is the minimum entry and
            M is the maximum entry in "numbers".
    """
    if fractions:
        return [sum(numbers == count) / len(numbers) for count in range(numbers.min(), numbers.max() + 1)]

    return [sum(numbers == count) for count in range(numbers.min(), numbers.max() + 1)]


def number_vertices(sections: list[ndarray[float]]) -> ndarray[int]:
    """
    Returns the number of vertices in a list of section profiles.
    :param sections: A list of length L, being the L section profiles. Here, entry i is a numpy.ndarray of shape (N_i,
            K), containing coordinates of all vertices in the i-th section profile. Here, N_i is the number of vertices
            in the i-th section profile, and K is the dimension of the vertices.
    :return: A numpy.ndarray of shape (M,), where entry i is equal to N_i, the number of vertices in the i-th section
            profile.
    """
    return np.array([len(sections[i]) for i in range(len(sections))], dtype=int)
