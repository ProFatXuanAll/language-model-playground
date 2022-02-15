"""Whitespace tokenizer class."""

import re
from typing import ClassVar, Final, List

from lmp.tknzr._base import SP_TKS, BaseTknzr

SPLIT_PTTN: Final[re.Pattern] = re.compile('(' + '|'.join(map(re.escape, SP_TKS)) + r'|\s+' + ')')


class WsTknzr(BaseTknzr):
  """Whitespace tokenizer class.

  Tokenize text into whitespaces seperated tokens.  No whitespace will be preserved after tokenization.

  Parameters
  ----------
  is_uncased: bool
    Set to ``True`` to convert text into lower cases.
  max_vocab: int
    Maximum vocabulary size.
  max_seq_len: int
    Automatically truncate or pad token id list to have length equal to ``max_seq_len``.
  min_count: int
    Minimum token occurrence counts.
  kwargs: typing.Any, optional
    Useless parameter.  Intently left for subclasses inheritance.

  Attributes
  ----------
  tknzr_name: typing.ClassVar[str]
    CLI name of whitespace tokenizer is ``whitespace``.

  See Also
  --------
  :doc:`lmp.tknzr </tknzr/index>`
    All available tokenizers.
  lmp.tknzr.BaseTknzr
    Tokenizer utilities.

  Examples
  --------
  >>> from lmp.tknzr import WsTknzr
  >>> tknzr = WsTknzr(is_uncased=False, max_seq_len=128, max_vocab=10, min_count=2)
  >>> assert tknzr.tknz('a b c') == ['a', 'b', 'c']
  >>> assert tknzr.dtknz(['a', 'b', 'c']) == 'a b c'
  """

  tknzr_name: ClassVar[str] = 'whitespace'

  def tknz(self, txt: str) -> List[str]:
    """Split text between whitespaces.

    Text will first be normalized by :py:meth:`lmp.tknzr.BaseTknz.norm`, then be splited between whitespaces.

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
    lmp.tknzr.WsTknzr.dtknz
      Join text with whitespaces.
    lmp.tknzr.BaseTknzr.norm
      Text normalization.

    Examples
    --------
    >>> from lmp.tknzr import WsTknzr
    >>> tknzr = WsTknzr(is_uncased=False, max_seq_len=128, max_vocab=10, min_count=2)
    >>> assert tknzr.tknz('a b c') == ['a', 'b', 'c']
    >>> assert tknzr.tknz('abc def') == ['abc', 'def']
    """
    # Perform normalization.
    txt = self.norm(txt)

    # First we split text with special token pattern, then we strip text to convert stand alone whitespaces into empty
    # string.  Finally we filter out empty string.
    return list(filter(bool, [tk.strip() for tk in SPLIT_PTTN.split(txt)]))

  def dtknz(self, tks: List[str]) -> str:
    """Join text with whitespaces.

    Insert whitespace between tokens.  Returned text is normalized by :py:meth:`lmp.tknzr.BaseTknz.norm`.

    Parameters
    ----------
    tks: list[str]
      token list to be joint.

    Returns
    -------
    str
      Normalized text with whitespaces in between.

    See Also
    --------
    lmp.tknzr.WsTknzr.tknz
      Split text between whitespaces.
    lmp.tknzr.BaseTknzr.norm
      Text normalization.

    Examples
    --------
    >>> from lmp.tknzr import WsTknzr
    >>> tknzr = WsTknzr(is_uncased=False, max_seq_len=128, max_vocab=10, min_count=2)
    >>> assert tknzr.dtknz(['a', 'b', 'c']) == 'a b c'
    >>> assert tknzr.dtknz(['abc', 'def']) == 'abc def'
    """
    # First perform detokenization, then do normalization.  Order of these operation does not affect the output.
    return self.norm(' '.join(tks))
