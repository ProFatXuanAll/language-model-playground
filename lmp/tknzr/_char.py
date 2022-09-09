"""Character tokenizer class.

Attributes
----------
SP_TKS_PTTN: typing.Final[re.Pattern]
  Special tokens matching pattern.
"""

import re
from typing import ClassVar, Final, List

from lmp.tknzr._base import BaseTknzr
from lmp.vars import SP_TKS

SP_TKS_PTTN: Final[re.Pattern] = re.compile('(' + '|'.join(map(re.escape, SP_TKS)) + ')')


class CharTknzr(BaseTknzr):
  """Character tokenizer class.

  Tokenize text into list of unicode characters.

  Parameters
  ----------
  is_uncased: bool, default: False
    Set to ``True`` to convert text into lowercase.
    Mainly used by :py:meth:`~norm`.
  max_vocab: int, default: -1
    Tokenizer's maximum vocabulary size.
    Set to ``-1`` to include as many unicode characters in vocabulary as possible.
    Mainly used by :py:meth:`~build_vocab`.
  min_count: int, default: 0
    Minimum character occurrence counts.
    Unicode characters have occurrence counts less than ``min_count`` will not be added to tokenizer's vocabulary.
    Mainly used by :py:meth:`~build_vocab`.
  kwargs: typing.Any, optional
    Useless parameter.
    Intently left for subclasses inheritance.

  Attributes
  ----------
  id2tk: dict[int, str]
    Character-to-id inverse lookup table.
  is_uncased: bool
    Convert text into lowercase if set to ``True``.
  max_vocab: int
    Tokenizer's maximum vocabulary size.
  min_count: int
    Minimum character occurrence counts.
  tk2id: dict[str, int]
    Character-to-id lookup table.
  tknzr_name: typing.ClassVar[str]
    CLI name of character tokenizer is ``character``.

  See Also
  --------
  :doc:`lmp.tknzr </tknzr/index>`
    All available tokenizers.

  Examples
  --------
  >>> from lmp.tknzr import CharTknzr
  >>> tknzr = CharTknzr()
  >>> assert tknzr.tknz('abc') == ['a', 'b', 'c']
  >>> assert tknzr.dtknz(['a', 'b', 'c']) == 'abc'
  """

  tknzr_name: ClassVar[str] = 'character'

  def tknz(self, txt: str) -> List[str]:
    """Convert text into character list.

    Text is first normalized then splitted into unicode character list.
    Each special token is treated as an unicode character and thus is not splitted.

    Parameters
    ----------
    txt: str
      Text to be tokenized.

    Returns
    -------
    list[str]
      List of normalized unicode characters.

    See Also
    --------
    ~dtknz
      Convert unicode character list back to text.
    ~norm
      Text normalization.

    Examples
    --------
    >>> from lmp.tknzr import CharTknzr
    >>> tknzr = CharTknzr()
    >>> assert tknzr.tknz('abc') == ['a', 'b', 'c']
    >>> assert tknzr.tknz('abc def') == ['a', 'b', 'c', ' ', 'd', 'e', 'f']
    """
    # Perform normalization.
    txt = self.norm(txt)

    # Perform tokenization.
    tks = []
    while txt:
      match = SP_TKS_PTTN.match(txt)
      if match:
        tks.append(match.group(1))
        txt = txt[len(tks[-1]):]
      else:
        tks.append(txt[0])
        txt = txt[1:]

    return tks

  def dtknz(self, tks: List[str]) -> str:
    """Convert unicode character list back to text.

    Unicode character list is joined without whitespaces.
    Returned text is normalized.

    Parameters
    ----------
    tks: list[str]
      Unicode character list to be joint.

    Returns
    -------
    str
      Normalized text without additional whitespaces other than the ones in the unicode character list.

    See Also
    --------
    ~tknz
      Convert text into unicode characters.
    ~norm
      Text normalization.

    Examples
    --------
    >>> from lmp.tknzr import CharTknzr
    >>> tknzr = CharTknzr()
    >>> assert tknzr.dtknz(['a', 'b', 'c']) == 'abc'
    >>> assert tknzr.dtknz(['a', 'b', 'c', ' ', 'd', 'e', 'f']) == 'abc def'
    """
    # First perform detokenization, then do normalization.
    # Order of these operation does not affect the output.
    return self.norm(''.join(tks))
