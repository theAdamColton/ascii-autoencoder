"""
Utils for generating one hot encodings for ascii characters,
only allowed characters are [32-126]
"""

import numpy as np
import torch

def get_one_hot_for_str(a: str) -> np.ndarray:
    out = np.zeros((len(a), 95))
    for i in range(len(a)):
        out[i] = get_one_hot_for_char(a[i])
    return out

def get_one_hot_for_char(a: str) -> np.ndarray:
    """a: character that is [32-126]"""
    code = ord(a) - 32
    vec = np.zeros(95)
    vec[code] = 1
    return vec

def fuzzy_one_hot_to_str(x: np.ndarray) -> str:
    """x: length(s) by 95 ndarray,
    x can have any floating point numbers between 0 and 1,
    and will be thresholded to produce a one hot vec based on
    the maximum value in each of its rows"""
    one_hot = np.zeros_like(x)
    one_hot[np.arange(len(x)), x.argmax(1)] = 1

    return one_hot_to_str(one_hot)


def one_hot_to_str(x: np.ndarray) -> str:
    """x: length(s) by 95 ndarray,
    with one hot encoding in each of its x[:] elements
    """
    x = x.reshape(-1, 95)
    out = ""
    for vec in x:
        out += one_hot_vec_to_char(vec)
    return out

def one_hot_vec_to_char(x: np.ndarray) -> str:
    """x: 95 array"""
    return chr(x.argmax() + 32)
