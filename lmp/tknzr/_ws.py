"""Whitespace tokenizer class.

Attributes
----------
SPLIT_PTTN: typing.Final[re.Pattern]
  Special tokens and whitespaces matching pattern.
"""

import re
from typing import ClassVar, Final, List

from lmp.tknzr._base import BaseTknzr
from lmp.vars import SP_TKS

SPLIT_PTTN: Final[re.Pattern] = re.compile('(' + '|'.join(map(re.escape, SP_TKS)) + r'|\s+' + ')')


class WsTknzr(BaseTknzr):
  """Whitespace tokenizer class.

  Tokenize text into whitespaces seperated tokens.
  No whitespace will be preserved after tokenization.

  Parameters
  ----------
  is_uncased: bool, default: False
    Set to ``True`` to convert text into lowercase.
    Mainly used by :py:meth:`~norm`.
  max_vocab: int, default: -1
    Tokenizer's maximum vocabulary size.
    Set to ``-1`` to include as many tokens in vocabulary as possible.
    Mainly used by :py:meth:`~build_vocab`.
  min_count: int, default: 0
    Minimum token occurrence counts.
    Tokens have occurrence counts less than ``min_count`` will not be added to tokenizer's vocabulary.
    Mainly used by :py:meth:`~build_vocab`.
  kwargs: typing.Any, optional
    Useless parameter.
    Intently left for subclasses inheritance.

  Attributes
  ----------
  id2tk: dict[int, str]
    Token-to-id inverse lookup table.
  is_uncased: bool
    Convert text into lowercase if set to ``True``.
  max_vocab: int
    Tokenizer's maximum vocabulary size.
  min_count: int
    Minimum token occurrence counts.
  tk2id: dict[str, int]
    Token-to-id lookup table.
  tknzr_name: typing.ClassVar[str]
    CLI name of whitespace tokenizer is ``whitespace``.

  See Also
  --------
  :doc:`lmp.tknzr </tknzr/index>`
    All available tokenizers.

  Examples
  --------
  >>> from lmp.tknzr import WsTknzr
  >>> tknzr = WsTknzr()
  >>> assert tknzr.tknz('a b c') == ['a', 'b', 'c']
  >>> assert tknzr.dtknz(['a', 'b', 'c']) == 'a b c'
  """

  tknzr_name: ClassVar[str] = 'whitespace'

  def tknz(self, txt: str) -> List[str]:
    """Split text on whitespaces.

    Text is first normalized then splited on whitespaces.

    Parameters
    ----------
    txt: str
      Text to be tokenized.

    Returns
    -------
    list[str]
      List of normalized whitespace-separated tokens.

    See Also
    --------
    ~dtknz
      Join text with whitespaces.
    ~norm
      Text normalization.

    Examples
    --------
    >>> from lmp.tknzr import WsTknzr
    >>> tknzr = WsTknzr()
    >>> assert tknzr.tknz('a b c') == ['a', 'b', 'c']
    >>> assert tknzr.tknz('abc def') == ['abc', 'def']
    """
    # Perform normalization.
    txt = self.norm(txt)

    # First we split text using special token pattern.
    # Then we strip text to convert stand alone whitespace into empty string.
    # Finally we filter out empty string.
    return list(filter(bool, [tk.strip() for tk in SPLIT_PTTN.split(txt)]))

  def dtknz(self, tks: List[str]) -> str:
    """Join tokens with whitespaces.

    Insert whitespace between tokens.
    Returned text is normalized.

    Parameters
    ----------
    tks: list[str]
      Token list to be joint.

    Returns
    -------
    str
      Normalized text with whitespaces in between.

    See Also
    --------
    ~tknz
      Split text on whitespaces.
    ~norm
      Text normalization.

    Examples
    --------
    >>> from lmp.tknzr import WsTknzr
    >>> tknzr = WsTknzr()
    >>> assert tknzr.dtknz(['a', 'b', 'c']) == 'a b c'
    >>> assert tknzr.dtknz(['abc', 'def']) == 'abc def'
    """
    # First perform detokenization, then do normalization.
    # Order of these operation does not affect the output.
    return self.norm(' '.join(tks))
