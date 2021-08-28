r"""Whitespace :term:`tokenizer` class."""

import re
from typing import ClassVar, List, Sequence

from lmp.tknzr._base import BaseTknzr


class WsTknzr(BaseTknzr):
    r"""Whitespace :term:`tokenizer` class.

    Tokenize text into (unicode) whitespace seperate tokens.
    No whitespace will be preserved after tokenization.

    Parameters
    ==========
    is_uncased: bool
        Convert text into lowercase if set to ``True``.
    max_vocab: int
        Maximum vocabulary size.
        Set to ``-1`` to include as many tokens as possible in vocabulary.
    min_count: int
        Minimum token frequency for each token to be included in tokenizer's
        vocabulary.
    tk2id: Dict[str, int], optional
        Token (a string) to id (an integer) lookup table.
        If ``tk2id is not None``, then initialize lookup table with ``tk2id``.
        Otherwise initialize lookup table with special tokens only.
    kwargs: Dict, optional
        Useless parameter.
        Intently left for subclass parameters extension.

    Attributes
    ==========
    tknzr_name: ClassVar[str]
        Tokenizer name is ``whitespace``.
        Used for command line argument parsing.

    Raises
    ======
    TypeError
        When parameters do not obey their type annotations.

    See Also
    ========
    lmp.tknzr.BaseTknzr

    Examples
    ========
    >>> from typing import List, Sequence
    >>> from lmp.tknzr import WsTknzr
    >>> tknzr = WsTknzr(is_uncased=False, max_vocab=10, min_count=2)
    >>> tknzr.tknz('a b c')
    ['a', 'b', 'c']
    >>> tknzr.dtknz(['a', 'b', 'c'])
    'a b c'
    """
    tknzr_name: ClassVar[str] = 'whitespace'

    def tknz(self, txt: str) -> List[str]:
        r"""Perform whitespace :term:`tokenization` on text.

        Text will first be normalized by :py:meth:`lmp.tknzr.BaseTknz.norm`,
        then be tokenized by whitespaces.

        Parameters
        ==========
        txt: str
            Text to be tokenized.

        Returns
        =======
        List[str]
            List of normalized whitespace-separated tokens.
            No whitespaces will be preserved after tokenization.

        See Also
        ========
        lmp.tknzr.WsTknzr.dtknz
        lmp.tknzr.BaseTknzr.norm

        Examples
        ========
        >>> from lmp.tknzr import WsTknzr
        >>> tknzr = WsTknzr(is_uncased=False, max_vocab=10, min_count=2)
        >>> tknzr.tknz('a b c')
        ['a', 'b', 'c']
        >>> tknzr.tknz('abc def')
        ['abc', 'def']
        """
        # First do normalization, then perform tokenization.
        tks = re.split(r'\s+', self.norm(txt))

        # Return empty list when `txt` is empty string.
        # This is needed since `re.split(r'\s+', '')` return `['']` instead of
        # `[]`.
        if tks == ['']:
            return []
        return tks

    def dtknz(self, tks: Sequence[str]) -> str:
        r"""Convert :term:`tokens` back to text.

        Tokens will be joined with one whitespace.
        Returned text is normalized by :py:meth:`lmp.tknzr.BaseTknz.norm`.

        Parameters
        ==========
        tks: Seqeuence[str]
            Sequence of tokens to be detokenized.

        Returns
        =======
        str
            Normalized text with whitespaces in between each token.

        See Also
        ========
        lmp.tknzr.WsTknzr.tknz
        lmp.tknzr.BaseTknzr.norm

        Examples
        ========
        >>> from lmp.tknzr import WsTknzr
        >>> tknzr = WsTknzr(is_uncased=False, max_vocab=10, min_count=2)
        >>> tknzr.dtknz(['a', 'b', 'c'])
        'a b c'
        >>> tknzr.dtknz(['abc', 'def'])
        'abc def'
        """
        # First perform detokenization, then do normalization.
        return self.norm(' '.join(tks))
