""" For compatability purposes this contains string manipulation methods that were added in python 3.9 """


def ljust(x: str, maxlen: int, char: str) -> str:
    diff = maxlen - len(x)
    return x + char * diff


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]


def remove_suffix(text, suffix) -> str:
    if text.endswith(suffix):
        return text[:len(suffix)]
    else: return text
