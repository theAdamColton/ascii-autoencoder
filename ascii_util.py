"""
Utilities for dealing with ascii art
"""

import numpy as np

from character_embeddings import one_hot_encoding
from string_utils import remove_prefix, remove_suffix, ljust

def raw_string_to_squareized(s: str, x: int) -> str:
    """
    Takes a string s, with max line length less than x, and number of lines less than x,
    and returns a x by x string with s in the middle of it, padded with space characters.
    """
    s_padded = pad_to_max_line_length(s)
    s_squareized = pad_to_x_by_x(s_padded, x)
    return s_squareized


def pad_to_max_line_length(s: str, char=" ") -> str:
    """Pads each line of s to the max line length of all the lines in s.
    char: character to pad with
    """
    maxlen = 0
    for l in s.splitlines():
        length = len(l)
        if length > maxlen:
            maxlen = length

    out = ""
    for l in s.splitlines():
        # Gets rid of the last '\n'
        line = remove_suffix(l, '\n')
        padded_line = ljust(line, maxlen, char)
        out += padded_line + "\n"

    return out


def pad_to_x_by_x(s: str, x: int, char=" ") -> str:
    """
    Pads ascii by centering it with ' ' chars
    Assumes that each line of ascii is already padded to the max
    length of its lines
    """
    lines = s.splitlines()
    line_width = len(lines[0])
    assert line_width <= x

    # Vertical padding
    total_vert_padding = x - len(lines)
    assert total_vert_padding >= 0
    assert total_vert_padding <= x
    toppad = total_vert_padding // 2
    botpad = total_vert_padding - toppad

    out = ""
    if toppad != 0:
        out = (vertical_pad(x, toppad, char=char) + "\n")
    out += "".join(
        line.replace("\n", "").center(x, char) + "\n"
        for line in lines
    )
    if botpad != 0:
        out += vertical_pad(x, botpad, char=char)

    return out


def vertical_pad(width: int, height: int, char=" ") -> str:
    if height == 0:
        return ""
    out = char*width
    out += ("\n" + char * width) * (height-1)
    return out


def string_reshape(s: str, x: int) -> str:
    """Adds line breaks to s so it becomes a square, x by x string"""
    assert len(s) % x == 0
    res = '\n'.join(s[i:i+x] for i in range(0, len(s), x))
    return res


def horizontal_concat(s1: str, s2: str, separator = "   |   ") -> str:
    """Concats two 'square' shaped strings"""
    out = ""
    for i, (line1, line2) in enumerate(zip(s1.split("\n"), s2.split("\n"))):
        if i == len(line1) -1:
            out += line1 + separator + line2
        else:
            out += line1 + separator + line2 + "\n"
    return out


"""                Numpy operations                         """

def raw_string_to_one_hot(s: str, x: int) -> np.ndarray:
    squareized_s = raw_string_to_squareized(s, x)
    return squareized_string_to_one_hot(squareized_s, x)

def squareized_string_to_one_hot(s: str, x: int) -> np.ndarray:
    """Takes a squareized string s and a length x,
    returns an 95 by x by x array of one hot encodings"""
    s = s.replace('\n', '')
    embedded = one_hot_encoding.get_one_hot_for_str(s)  
    embedded = embedded.reshape(x, x, 95)
    # Makes embeddings nchannels by image_res by image_res
    embedded = np.moveaxis(embedded, 2,0)
    return embedded

def one_hot_embedded_matrix_to_string(a: np.ndarray) -> str:
    """Takes a 95 by x by x matrix a of one hot character embeddings and
    returns a string"""
    res = a.shape[1]
    # Moves channels to last dim
    a = np.moveaxis(a, 0, 2)
    # Flattens
    a = a.reshape(res**2, 95)
    flat_s = one_hot_encoding.fuzzy_one_hot_to_str(a)
    squareized_s = string_reshape(flat_s, res)
    return squareized_s

