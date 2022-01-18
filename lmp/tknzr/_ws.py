"""Whitespace tokenizer class."""

import re
from typing import ClassVar, List

from lmp.tknzr._base import BaseTknzr


class WsTknzr(BaseTknzr):
  """Whitespace tokenizer class.

  Tokenize text into whitespaces seperated tokens.  No whitespace will be preserved after tokenization.

  Parameters
  ----------
  is_uncased: bool
    Set to ``True`` to convert text into lower cases.
  max_vocab: int
    Maximum vocabulary size.
  min_count: int
    Minimum token occurrence counts.
  tk2id: dict[str, int], default: None
    Token-to-id lookup table.
  kwargs: dict, optional
    Useless parameter.  Intently left for subclass inheritance.

  Attributes
  ----------
  tknzr_name: typing.ClassVar[str]
    Whitespace tokenizer's name is ``whitespace``.

  See Also
  --------
  lmp.tknzr
    All available tokenizers.
  lmp.tknzr.BaseTknzr
    Tokenizer utilities.

  Examples
  --------
  >>> from lmp.tknzr import WsTknzr
  >>> tknzr = WsTknzr(is_uncased=False, max_vocab=10, min_count=2)
  >>> tknzr.tknz('a b c')
  ['a', 'b', 'c']
  >>> tknzr.dtknz(['a', 'b', 'c'])
  'a b c'
  """

  tknzr_name: ClassVar[str] = 'whitespace'

  def tknz(self, txt: str) -> List[str]:
    """Split text between whitespaces.

    Text will first be normalized by :py:meth:`lmp.tknzr.BaseTknz.norm`,  then be splited between whitespaces.

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
    >>> tknzr = WsTknzr(is_uncased=False, max_vocab=10, min_count=2)
    >>> tknzr.tknz('a b c')
    ['a', 'b', 'c']
    >>> tknzr.tknz('abc def')
    ['abc', 'def']
    """
    # Perform normalization.
    txt = self.norm(txt)

    # Whitespaces and special tokens recognizer.
    ptn = re.compile(
      '(' + '|'.join(
        map(
          re.escape,
          [
            self.__class__.bos_tk,
            self.__class__.eos_tk,
            self.__class__.pad_tk,
            self.__class__.unk_tk,
          ],
        )
      ) + r'|\s+' + ')'
    )

    # First we split text with pattern defined above, then we strip text to convert stand alone whitespaces into empty
    # string.  Finally we filter out empty string.
    return list(filter(bool, [tk.strip() for tk in ptn.split(txt)]))

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
    >>> tknzr = WsTknzr(is_uncased=False, max_vocab=10, min_count=2)
    >>> tknzr.dtknz(['a', 'b', 'c'])
    'a b c'
    >>> tknzr.dtknz(['abc', 'def'])
    'abc def'
    """
    # First perform detokenization, then do normalization.  Order of these operation does not affect the output.
    return self.norm(' '.join(tks))
