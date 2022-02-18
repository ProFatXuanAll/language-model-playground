"""Character tokenizer class.

Attributes
----------
SP_TKS_PTTN: typing.Final[re.Pattern]
  Special tokens matching pattern.
"""

import argparse
import re
from typing import ClassVar, Final, List

from lmp.tknzr._base import SP_TKS, BaseTknzr

SP_TKS_PTTN: Final[re.Pattern] = re.compile('(' + '|'.join(map(re.escape, SP_TKS)) + ')')


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
  kwargs: typing.Any, optional
    Useless parameter.  Intently left for subclasses inheritance.

  Attributes
  ----------
  tknzr_name: typing.ClassVar[str]
    CLI name of character tokenizer is ``character``.

  See Also
  --------
  :doc:`lmp.tknzr </tknzr/index>`
    All available tokenizers.
  lmp.tknzr.BaseTknzr
    Tokenizer utilities.

  Examples
  --------
  >>> from lmp.tknzr import CharTknzr
  >>> tknzr = CharTknzr(is_uncased=False, max_vocab=10, min_count=2)
  >>> assert tknzr.tknz('abc') == ['a', 'b', 'c']
  >>> assert tknzr.dtknz(['a', 'b', 'c']) == 'abc'
  """

  tknzr_name: ClassVar[str] = 'character'

  @classmethod
  def add_CLI_args(cls, parser: argparse.ArgumentParser) -> None:
    """Add character tokenizer constructor parameters to CLI arguments parser.

    Parameters
    ----------
    parser: argparse.ArgumentParser
      CLI arguments parser.

    Returns
    -------
    None

    See Also
    --------
    :doc:`lmp.script.train_tknzr </script/train_tknzr>`
      Tokenizer training script.

    Examples
    --------
    >>> import argparse
    >>> from lmp.tknzr import CharTknzr
    >>> parser = argparse.ArgumentParser()
    >>> CharTknzr.add_CLI_args(parser)
    >>> args = parser.parse_args([
    ...   '--max_vocab', '10',
    ...   '--min_count', '2',
    ... ])
    >>> assert args.is_uncased == False
    >>> assert args.max_vocab == 10
    >>> assert args.min_count == 2
    """
    super().add_CLI_args(parser=parser)

    # Required arguments.
    group = parser.add_argument_group('character tokenizer constructor arguments')
    group.add_argument(
      '--max_vocab',
      help='Maximum vocabulary size.  Set to `-1` to include any tokens.',
      required=True,
      type=int,
    )
    group.add_argument(
      '--min_count',
      help='Minimum token occurrence count.  Set to `0` to disable.',
      required=True,
      type=int,
    )

    # Optional arguments.
    group.add_argument(
      '--is_uncased',
      action='store_true',
      help='Convert all text and tokens into lower cases if given.',
    )

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
    >>> assert tknzr.dtknz(['a', 'b', 'c']) == 'abc'
    >>> assert tknzr.dtknz(['a', 'b', 'c', ' ', 'd', 'e', 'f']) == 'abc def'
    """
    # First perform detokenization, then do normalization.  Order of these operation does not affect the output.
    return self.norm(''.join(tks))
