import re
import unicodedata


def norm(seq: str) -> str:
    r"""Perform normalization on input sequence.

    Input sequence will first be _NFKC normalized, then convert consecutive
    whitespaces into single whitespace. Both leading and trailing whitespaces
    will be stripped.

    Parameters
    ==========
    seq : str
        Input sequence to be normalized.

    Returns
    =======
    str
        Normalized sequence.

    References
    ==========
    .. _NFKC: https://en.wikipedia.org/wiki/Unicode_equivalence
    """
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', seq).strip())
