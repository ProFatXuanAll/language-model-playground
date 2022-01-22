"""Character tokenizer class."""

import re
from typing import ClassVar, List

from lmp.tknzr._base import BaseTknzr


class CharTknzr(BaseTknzr):
  """Character tokenizer class.

  Tokenize text into list of characters.

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
  kwargs: typing.Any, optional
    Useless parameter.  Intently left for subclasses inheritance.

  Attributes
  ----------
  tknzr_name: typing.ClassVar[str]
    CLI name of character tokenizer is ``character``.

  See Also
  --------
  lmp.tknzr
    All available tokenizers.
  lmp.tknzr.BaseTknzr
    Tokenizer utilities.

  Examples
  --------
  >>> from lmp.tknzr import CharTknzr
  >>> tknzr = CharTknzr(is_uncased=False, max_vocab=10, min_count=2)
  >>> tknzr.tknz('abc')
  ['a', 'b', 'c']
  >>> tknzr.dtknz(['a', 'b', 'c'])
  'abc'
  """

  tknzr_name: ClassVar[str] = 'character'

  def tknz(self, txt: str) -> List[str]:
    """Convert text into list of characters.

    Text will first be normalized by :py:meth:`lmp.tknzr.BaseTknz.norm`, then be splitted into list of characters.
    Special tokens will not be splitted.

    Parameters
    ----------
    txt: str
      Text to be tokenized.

    Returns
    -------
    list[str]
      List of normalized characters.

    See Also
    --------
    lmp.tknzr.CharTknzr.dtknz
      Join characters back to text.
    lmp.tknzr.BaseTknzr.norm
      Text normalization.

    Examples
    --------
    >>> from lmp.tknzr import CharTknzr
    >>> tknzr = CharTknzr(is_uncased=False, max_vocab=10, min_count=2)
    >>> tknzr.tknz('abc')
    ['a', 'b', 'c']
    >>> tknzr.tknz('abc def')
    ['a', 'b', 'c', ' ', 'd', 'e', 'f']
    """
    # Perform normalization.
    txt = self.norm(txt)

    # Special tokens recognizer.
    sp_tks_ptn = re.compile(
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
      ) + ')'
    )

    # Perform tokenization.
    tks = []
    while txt:
      match = sp_tks_ptn.match(txt)
      if match:
        tks.append(match.group(1))
        txt = txt[len(tks[-1]):]
      else:
        tks.append(txt[0])
        txt = txt[1:]

    return tks

  def dtknz(self, tks: List[str]) -> str:
    """Join list of characters back to text.

    Tokens will be joined without whitespaces.  Returned text is normalized by :py:meth:`lmp.tknzr.BaseTknz.norm`.

    Parameters
    ----------
    tks: list[str]
      Token list to be joint.

    Returns
    -------
    str
      Normalized text without additional whitespaces other than the ones in the token list.

    See Also
    --------
    lmp.tknzr.CharTknzr.tknz
      Convert text into characters.
    lmp.tknzr.BaseTknzr.norm
      Text normalization.

    Examples
    --------
    >>> from lmp.tknzr import CharTknzr
    >>> tknzr = CharTknzr(is_uncased=False, max_vocab=10, min_count=2)
    >>> tknzr.dtknz(['a', 'b', 'c'])
    'abc'
    >>> tknzr.dtknz(['a', 'b', 'c', ' ', 'd', 'e', 'f'])
    'abc def'
    """
    # First perform detokenization, then do normalization.  Order of these operation does not affect the output.
    return self.norm(''.join(tks))
