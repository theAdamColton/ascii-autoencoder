"""
Utilities for dealing with ascii art
"""


def pad_to_max_line_length(s: str, char=" ") -> str:
    """Pads each line of s to the max line length of all the lines in s.
    char: character to pad with
    """
    maxlen = 0
    for l in s.splitlines():
        length = len(l)
        if length > maxlen:
            maxlen = length

    print(maxlen)
    
    out = ""
    for l in s.splitlines():
        # Gets rid of the last '\n'
        line = l.removesuffix('\n')
        padded_line = line.ljust(maxlen, char)
        out += padded_line + "\n"

    return out


def pad_to_x_by_x(ascii: str, x: int, char=" ") -> str:
    """
    Pads ascii by centering it with ' ' chars
    Assumes that each line of ascii is already padded to the max
    length of its lines
    """
    lines = ascii.splitlines()
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
