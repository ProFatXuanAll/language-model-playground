"""Tokenizer base class and utilities.

Attributes
----------
WS_PTTN: typing.Final[re.Pattern]
  Whitespace matching pattern.
"""

import abc
import argparse
import re
import typing
import unicodedata
from collections import Counter
from typing import Any, ClassVar, Dict, Final, Iterable, List

import lmp.util.validate
from lmp.vars import BOS_TK, BOS_TKID, EOS_TK, EOS_TKID, PAD_TK, PAD_TKID, UNK_TK, UNK_TKID

WS_PTTN: Final[re.Pattern] = re.compile(r'\s+')


class BaseTknzr(abc.ABC):
  """Tokenizer abstract base class.

  Provide text processing functionalities including tokenization, normalization and language model training formation.

  This class is designed to be the abstract base class of all tokenizers.
  Both tokenization and detokenization are left unimplemented.

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
    CLI name of the tokenizer.
    Only used to parse CLI arguments.

  See Also
  --------
  :doc:`lmp.tknzr </tknzr/index>`
    All available tokenizers.
  """

  tknzr_name: ClassVar[str] = 'base'

  def __init__(
    self,
    *,
    is_uncased: bool = False,
    max_vocab: int = -1,
    min_count: int = 0,
    **kwargs: Any,
  ):
    # `is_uncased` validation.
    lmp.util.validate.raise_if_not_instance(val=is_uncased, val_name='is_uncased', val_type=bool)
    self.is_uncased = is_uncased

    # `max_vocab` validation.
    lmp.util.validate.raise_if_not_instance(val=max_vocab, val_name='max_vocab', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[-1, max_vocab], val_names=['-1', 'max_vocab'])
    self.max_vocab = max_vocab

    # `min_count` validation.
    lmp.util.validate.raise_if_not_instance(val=min_count, val_name='min_count', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[0, min_count], val_names=['0', 'min_count'])
    self.min_count = min_count

    self.tk2id: Dict[str, int] = {}
    self.id2tk: Dict[int, str] = {}

    # Initialize vocabulary with special tokens.
    for tk, tkid in [(BOS_TK, BOS_TKID), (EOS_TK, EOS_TKID), (PAD_TK, PAD_TKID), (UNK_TK, UNK_TKID)]:
      self.tk2id[tk] = tkid
      self.id2tk[tkid] = tk

  @classmethod
  def add_CLI_args(cls, parser: argparse.ArgumentParser) -> None:
    """Add tokenizer hyperparameters to CLI argument parser.

    Parameters
    ----------
    parser: argparse.ArgumentParser
      CLI argument parser.

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
    >>> from lmp.tknzr import BaseTknzr
    >>> parser = argparse.ArgumentParser()
    >>> BaseTknzr.add_CLI_args(parser)
    >>> args = parser.parse_args([
    ...   '--max_vocab', '10',
    ...   '--min_count', '2',
    ... ])
    >>> assert args.is_uncased == False
    >>> assert args.max_vocab == 10
    >>> assert args.min_count == 2
    """
    # `parser` validation.
    lmp.util.validate.raise_if_not_instance(val=parser, val_name='parser', val_type=argparse.ArgumentParser)

    # Add tokenizer hyperparameters to CLI arguments.
    group = parser.add_argument_group('Tokenizer hyperparameters')
    group.add_argument(
      '--max_vocab',
      default=-1,
      help='''
      Maximum vocabulary size.
      Set to `-1` to include any tokens.
      Default is `-1`.
      ''',
      type=int,
    )
    group.add_argument(
      '--min_count',
      default=0,
      help='''
      Minimum token occurrence count for a token to be included in tokenizer's vocabulary.
      Set to `0` to disable.
      Default is `0`.
      ''',
      type=int,
    )
    group.add_argument(
      '--is_uncased',
      action='store_true',
      help='''
      Convert all text and tokens into lowercase if given.
      Default is ``False``.
      ''',
    )

  def build_vocab(self, batch_txt: Iterable[str]) -> None:
    """Build tokenizer's vocabulary.

    Build vocabulary based on token occurrence counts.
    Text in ``batch_txt`` is first normalized and tokenized, then count each token's occurrence.
    Tokens with higher occurrence counts are added to vocabulary first.
    Tokens with the same occurrence counts are added to vocabulary in the order of their appearance.

    When adding a new token to vocabulary, its token id will be assign to the largest token id + 1.
    Tokens already in vocabulary are not added to vocabulary again.
    If a token's occurrence count is lower than ``self.min_count``, then that token is not added to vocabulary.
    If vocabulary size is larger than or equal to ``self.max_vocab``, then no new tokens are added to vocabulary.

    Parameters
    ----------
    batch_txt: collections.abc.Iterable[str]
      Source of text to build vocabulary.

    Returns
    -------
    None

    See Also
    --------
    ~norm
      Perform normalization on text.
    ~tknz
      Perform tokenization on text.
    ~vocab_size
      Tokenizer's vocabulary size.
    """
    # `batch_txt` validation.
    lmp.util.validate.raise_if_not_instance(val=batch_txt, val_name='batch_txt', val_type=Iterable)

    # Count each token's occurrence.
    c: typing.Counter[str] = Counter()
    for txt in batch_txt:
      c.update(self.tknz(txt=self.norm(txt=txt)))

    max_id = max(self.tk2id.values()) + 1
    for tk, tk_count in c.most_common():
      # Stop adding tokens when pass vocabulary size limit.
      # Add as many tokens into vocabulary as possible when `self.max_vocab == -1`.
      if self.max_vocab != -1 and max_id >= self.max_vocab:
        break

      # Stop adding the token when the token occurrence count is low.
      # Since we sort token by occurrence count, tokens in the remaining loops will not have occurrence count higher
      # than `self.min_count` and thus we can break loop safely.
      if tk_count < self.min_count:
        break

      # Skip token if that token is already in vocabulary.
      if tk in self.tk2id:
        continue

      # Add new token to vocabulary.
      self.tk2id[tk] = max_id
      self.id2tk[max_id] = tk

      # Increment token id.
      max_id += 1

  def dec(self, tkids: List[int], *, rm_sp_tks: bool = False) -> str:
    """Decode token id list back to text.

    Token id list is first converted into token list then detokenized back to text.
    Special tokens other than ``<unk>`` will be removed if setting ``rm_sp_tks=True``.
    Token ids not in tokenizer's inverse lookup table are converted into ``<unk>`` token.

    Parameters
    ----------
    tkids: list[int]
      Token id list to be decoded.
    rm_sp_tks: bool, default: False
      Set to ``True`` to remove ``<bos>``, ``<eos>`` and ``<pad>``.

    Returns
    -------
    str
      Decoded text.

    See Also
    --------
    ~dtknz
      Convert tokens back to text.
    ~enc
      Encode text into token id list.

    Note
    ----
    Unknown tokens ``<unk>`` will not be removed even if setting ``rm_sp_tks=True``.
    This is simply because we do not know which token to convert it back (thus the name *unknown token*).
    """
    # `tkids` validation.
    lmp.util.validate.raise_if_not_instance(val=tkids, val_name='tkids', val_type=list)
    # `rm_sp_tks` validation.
    lmp.util.validate.raise_if_not_instance(val=rm_sp_tks, val_name='rm_sp_tks', val_type=bool)

    # Remove special token ids.
    if rm_sp_tks:
      sp_tkids = [BOS_TKID, EOS_TKID, PAD_TKID]
      tkids = list(filter(lambda tkid: tkid not in sp_tkids, tkids))

    tks = []
    # Convert token ids into tokens.
    for tkid in tkids:
      try:
        tks.append(self.id2tk[tkid])
      # Convert unknown token ids into `<unk>` token.
      except KeyError:
        tks.append(UNK_TK)

    return self.dtknz(tks)

  @abc.abstractmethod
  def dtknz(self, tks: List[str]) -> str:
    """Convert tokens back to text.

    Tokens will be detokenized and normalized by :py:meth:`~BaseTknz.norm`.
    The execution order of detokenization and normalization will not effect the result.

    Parameters
    ----------
    tks: list[str]
      Token list to be detokenized.

    Returns
    -------
    str
      Text which is normalized and detokenized from token list.

    See Also
    --------
    ~tknz
      Tokenize text into token list.
    ~norm
      Text normalization.
    """
    raise NotImplementedError

  def enc(self, txt: str) -> List[int]:
    """Encode text into token id list.

    Text will be tokenized into token list (``tk_0, tk_1, ..., tk_n``) and formatted as follow::

      <bos> tk_0 tk_1 ... tk_n <eos>

    - ``<bos>`` is the "begin of sequence" token.
    - ``<eos>`` is the "end of sequence" token.
    - ``<unk>`` token is used to replace OOV tokens.

    All tokens in token list are converted into token ids and returned.

    Parameters
    ----------
    txt: str
      Text to be encoded.

    Returns
    -------
    list[int]
      Token ids list.

    See Also
    --------
    ~dec
      Decode token id list back to text.
    ~pad_to_max
      Pad token id list to specified length.
    ~tknz
      Perform tokenization on text.
    """
    # Prepend `<bos>` token id.
    tkids = [BOS_TKID]

    # Convert tokens into token ids.
    for tk in self.tknz(txt):
      # Perform token id lookup.
      try:
        tkids.append(self.tk2id[tk])
      # Convert unknown tokens into `<unk>` token id.
      except KeyError:
        tkids.append(UNK_TKID)

    # Append `<eos>` token id.
    tkids.append(EOS_TKID)

    return tkids

  def norm(self, txt: str) -> str:
    """Perform text normalization.

    Text are normalized by NFKC.
    Whitespaces are collapsed and stripped from both ends.
    Text are converted into lowercase if setting ``is_uncased=True``.

    Parameters
    ----------
    txt: str
      Text to be normalized.

    Returns
    -------
    str
      Normalized text.

    See Also
    --------
    unicodedata.normalize
      Python built-in unicode normalization.

    Examples
    --------
    Convert text to lowercase.

    >>> from lmp.tknzr import CharTknzr
    >>> tknzr = CharTknzr(is_uncased=True)
    >>> assert tknzr.norm('ABC') == 'abc'
    """
    # `txt` validation.
    lmp.util.validate.raise_if_not_instance(val=txt, val_name='txt', val_type=str)

    norm_txt = WS_PTTN.sub(' ', unicodedata.normalize('NFKC', txt)).strip()
    if self.is_uncased:
      return norm_txt.lower()
    return norm_txt

  def pad_to_max(self, max_seq_len: int, tkids: List[int]) -> List[int]:
    """Pad token id list to specified length.

    If ``len(tkids) < max_seq_len``, then append padding token id at the end of ``tkids`` until ``tkids`` has length
    equal to ``max_seq_len``.
    Do nothing when ``len(tkids) >= max_seq_len``.

    Arguments
    ---------
    max_seq_len: int
      Maximum length constraint.
    tkids: list[int]
      Token id list to be padded.

    Returns
    -------
    list[int]
      Padded token id list.

    Examples
    --------
    >>> from lmp.vars import PAD_TKID
    >>> from lmp.tknzr import CharTknzr
    >>> tknzr = CharTknzr()
    >>> assert tknzr.pad_to_max(max_seq_len=4, tkids=[1, 2, 3]) == [1, 2, 3, PAD_TKID]
    """
    # `max_seq_len` validation.
    lmp.util.validate.raise_if_not_instance(val=max_seq_len, val_name='max_seq_len', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, max_seq_len], val_names=['1', 'max_seq_len'])

    # Calculate padding length.
    pad_len = max(0, max_seq_len - len(tkids))

    # Pad to maximum sequence length.
    return tkids + [PAD_TKID] * pad_len

  @abc.abstractmethod
  def tknz(self, txt: str) -> List[str]:
    """Perform tokenization on text.

    Text is first normalized then tokenized into token list.

    Parameters
    ----------
    txt: str
      Text to be tokenized.

    Returns
    -------
    list[str]
      List of normalized tokens.

    See Also
    --------
    ~dtknz
      Detokenize token list back to text.
    ~norm
      Text normalization.
    """
    raise NotImplementedError

  @property
  def vocab_size(self) -> int:
    """Get tokenizer vocabulary size.

    Returns
    -------
    int
      Tokenizer vocabulary size.

    See Also
    --------
    ~build_vocab
      Build vocabulary for tokenizer.
    """
    return len(self.tk2id)
