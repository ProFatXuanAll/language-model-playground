import re
import unicodedata


def norm(seq: str) -> str:
    # NFKC normalization.
    # Convert consecutive whitespace characters into single whitespace
    # character. Stripping both leading and trailing whitespace characters.
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', seq).strip())
