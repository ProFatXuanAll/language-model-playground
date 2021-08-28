r"""Character :term:`tokenizer` class."""


from typing import ClassVar, List, Sequence

from lmp.tknzr._base import BaseTknzr


class CharTknzr(BaseTknzr):
    r"""Character :term:`tokenizer` class.

    Tokenize text into (unicode) character.

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
        Tokenizer name is ``character``.
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
    >>> from lmp.tknzr import CharTknzr
    >>> tknzr = CharTknzr(is_uncased=False, max_vocab=10, min_count=2)
    >>> tknzr.tknz('abc')
    ['a', 'b', 'c']
    >>> tknzr.dtknz(['a', 'b', 'c'])
    'abc'
    """
    tknzr_name: ClassVar[str] = 'character'

    def tknz(self, txt: str) -> List[str]:
        r"""Perform character :term:`tokenization` on text.

        Text will first be normalized by :py:meth:`lmp.tknzr.BaseTknz.norm`,
        then be tokenized into list of characters.

        Parameters
        ==========
        txt: str
            Text to be tokenized.

        Returns
        =======
        List[str]
            List of normalized characters.

        See Also
        ========
        lmp.tknzr.CharTknzr.dtknz
        lmp.tknzr.BaseTknzr.norm

        Examples
        ========
        >>> from lmp.tknzr import CharTknzr
        >>> tknzr = CharTknzr(is_uncased=False, max_vocab=10, min_count=2)
        >>> tknzr.tknz('abc')
        ['a', 'b', 'c']
        >>> tknzr.tknz('abc def')
        ['a', 'b', 'c', ' ', 'd', 'e', 'f']
        """
        # First do normalization, then perform tokenization.
        return list(self.norm(txt))

    def dtknz(self, tks: Sequence[str]) -> str:
        r"""Convert :term:`tokens` back to text.

        Tokens will be joined without whitespaces.
        Returned text is normalized by :py:meth:`lmp.tknzr.BaseTknz.norm`.

        Parameters
        ==========
        tks: Seqeuence[str]
            Sequence of tokens to be detokenized.

        Returns
        =======
        str
            Normalized text without additional whitespaces other than the ones
            came with tokens.

        See Also
        ========
        lmp.tknzr.CharTknzr.tknz
        lmp.tknzr.BaseTknzr.norm

        Examples
        ========
        >>> from lmp.tknzr import CharTknzr
        >>> tknzr = CharTknzr(is_uncased=False, max_vocab=10, min_count=2)
        >>> tknzr.dtknz(['a', 'b', 'c'])
        'abc'
        >>> tknzr.dtknz(['a', 'b', 'c', ' ', 'd', 'e', 'f'])
        'abc def'
        """
        # First perform detokenization, then do normalization.
        return self.norm(''.join(tks))
