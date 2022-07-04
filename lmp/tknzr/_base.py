"""Tokenizer base class and utilities.

Attributes
----------
BOS_TK: typing.Final[str]
  A special token which represents the begining of a text.
BOS_TKID: typing.Final[int]
  Token id of :py:attr:`lmp.tknzr._base.BOS_TK`.
EOS_TK: typing.Final[str]
  A special token which represents the end of a text.
EOS_TKID: typing.Final[int]
  Token id of :py:attr:`lmp.tknzr._base.EOS_TK`.
PAD_TK: typing.Final[str]
  A special token which represents paddings of a text.
PAD_TKID: typing.Final[int]
  Token id of :py:attr:`lmp.tknzr._base.PAD_TK`.
SP_TKS: typing.Final[list[str]]
  List of special tokens.
UNK_TK: typing.Final[str]
  A special token which represents unknown tokens in a text.
UNK_TKID: typing.Final[int]
  Token id of :py:attr:`lmp.tknzr._base.UNK_TK`.
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

import lmp.util.path
import lmp.util.validate

BOS_TK: Final[str] = '[bos]'
BOS_TKID: Final[int] = 0
EOS_TK: Final[str] = '[eos]'
EOS_TKID: Final[int] = 1
PAD_TK: Final[str] = '[pad]'
PAD_TKID: Final[int] = 2
UNK_TK: Final[str] = '[unk]'
UNK_TKID: Final[int] = 3
SP_TKS: Final[List[str]] = [BOS_TK, EOS_TK, PAD_TK, UNK_TK]
WS_PTTN: Final[re.Pattern] = re.compile(r'\s+')


class BaseTknzr(abc.ABC):
  """Tokenizer abstract base class.

  Implement basic functionalities for text processing, including text normalization, saving and loading tokenizer
  configurations.

  This class is designed to be the abstract base class of all tokenizers, thus both tokenization and detokenization
  functions are left unimplemented.

  Parameters
  ----------
  is_uncased: bool
    Set to ``True`` to convert text into lower cases.
    Mainly used by :py:meth:`lmp.tknzr.BaseTknzr.norm`.
  max_vocab: int
    Tokenizer's maximum vocabulary size.
    Set to ``-1`` to include as many tokens in vocabulary as possible.
    Mainly used by :py:meth:`lmp.tknzr.BaseTknzr.build_vocab`.
  min_count: int
    Minimum token occurrence counts.
    Tokens have occurrence counts less than ``min_count`` will not be added to tokenizer's vocabulary.
    Mainly used by :py:meth:`lmp.tknzr.BaseTknzr.build_vocab`.
  kwargs: typing.Any, optional
    Useless parameter.
    Intently left for subclasses inheritance.

  Attributes
  ----------
  id2tk: dict[int, str]
    Token-to-id inverse lookup table.
  is_uncased: bool
    Convert text into lower cases if set to ``True``.
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

  def __init__(self, *, is_uncased: bool, max_vocab: int, min_count: int, **kwargs: Any):
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
    """Add tokenizer constructor parameters to CLI arguments parser.

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
    """
    # `parser` validation.
    lmp.util.validate.raise_if_not_instance(val=parser, val_name='parser', val_type=argparse.ArgumentParser)

  def norm(self, txt: str) -> str:
    """Perform normalization on text.

    Text will be NFKC normalized.
    Whitespaces are collapsed and strip from both ends.
    If ``self.is_uncased == True``, then text will be converted to lower cases.

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
    Convert text to lower cases.

    >>> from lmp.tknzr import CharTknzr
    >>> tknzr = CharTknzr(is_uncased=True, max_vocab=10, min_count=2)
    >>> assert tknzr.norm('ABC') == 'abc'
    """
    # `txt` validation.
    lmp.util.validate.raise_if_not_instance(val=txt, val_name='txt', val_type=str)

    norm_txt = WS_PTTN.sub(' ', unicodedata.normalize('NFKC', txt)).strip()
    if self.is_uncased:
      return norm_txt.lower()
    return norm_txt

  @abc.abstractmethod
  def tknz(self, txt: str) -> List[str]:
    """Perform tokenization on text.

    Text will first be normalized by :py:meth:`lmp.tknzr.BaseTknz.norm`, then be tokenized into list of tokens.

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
    lmp.tknzr.BaseTknzr.dtknz
      Detokenize list of tokens back to text.
    lmp.tknzr.BaseTknzr.norm
      Text normalization.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def dtknz(self, tks: List[str]) -> str:
    """Convert tokens back to text.

    :term:`Tokens` will be detokenized and normalized by :py:meth:`lmp.tknzr.BaseTknz.norm`.
    The execution order of detokenization and normalization will not effect the result.

    Parameters
    ----------
    tks: list[str]
      List of tokens to be detokenized.

    Returns
    -------
    str
      Normalized text detokenized from tokens.

    See Also
    --------
    lmp.tknzr.BaseTknzr.tknz
      Tokenize text into list of tokens.
    lmp.tknzr.BaseTknzr.norm
      Text normalization.
    """
    raise NotImplementedError

  def build_vocab(self, batch_txt: Iterable[str]) -> None:
    """Build vocabulary for tokenizer.

    Build :term:`vocabulary` based on :term:`token` occurrence counts.
    Each text in ``batch_txt`` will first be normalized and tokenized.
    We then count each token's occurrence and build vocabulary based on occurrence count.
    Vocabulary will be sorted by token occurrence counts in descending order.

    When adding a new token to vocabulary, its token id will be assign to the largest token id + 1.
    Tokens already in vocabulary will not be added to vocabulary again.
    If a token's occurrence count is lower than ``self.min_count``, then that token will not be added to vocabulary.
    If the size of vocabulary is larger than or equal to ``self.max_vocab``, then no new tokens will be added to
    vocabulary.

    Parameters
    ----------
    batch_txt: collections.abc.Iterable[str]
      Source of text to build vocabulary.

    Returns
    -------
    None

    See Also
    --------
    lmp.tknzr.BaseTknzr.norm
      Perform normalization on text.
    lmp.tknzr.BaseTknzr.tknz
      Perform tokenization on text.
    lmp.tknzr.BaseTknzr.vocab_size
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
      # Since we sort token by occurrence count, tokens enumerated in the remaining loops will not have occurrence
      # count higher than `self.min_count` and thus we can break loop savely.
      if tk_count < self.min_count:
        break

      # Skip token if that token is already in the vocabulary.
      if tk in self.tk2id:
        continue

      # Add new token to vocabulary.
      self.tk2id[tk] = max_id
      self.id2tk[max_id] = tk

      # Increment token id.
      max_id += 1

  @property
  def vocab_size(self) -> int:
    """Get tokenizer vocabulary size.

    Returns
    -------
    int
      Tokenizer vocabulary size.

    See Also
    --------
    lmp.tknzr.BaseTknzr.build_vocab
      Build vocabulary for tokenizer.
    """
    return len(self.tk2id)

  def trunc_to_max(self, max_seq_len: int, tkids: List[int]) -> List[int]:
    """Truncate token id list when token id list is longer than allowed.

    If ``len(tkids) > max_seq_len``, then truncate ``tkids`` to have length equals to ``max_seq_len``.
    Do nothing when ``len(tkids) <= max_seq_len``.

    Arguments
    ---------
    max_seq_len: int
      Maximum length constraint.
    tkids: list[int]
      Token id list to be truncated.

    Returns
    -------
    list[int]
      Truncated token id list.

    See Also
    --------
    lmp.tknzr.BaseTknzr.pad_to_max
      Pad token id list when token id list is shorter than required.

    Examples
    --------
    >>> from lmp.tknzr import CharTknzr
    >>> tknzr = CharTknzr(is_uncased=True, max_vocab=10, min_count=1)
    >>> assert tknzr.trunc_to_max(max_seq_len=1, tkids=[1, 2, 3]) == [1]
    """
    # `max_seq_len` validation.
    lmp.util.validate.raise_if_not_instance(val=max_seq_len, val_name='max_seq_len', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, max_seq_len], val_names=['1', 'max_seq_len'])

    # Truncate token id list to maximum sequence length.
    return tkids[:max_seq_len]

  def pad_to_max(self, max_seq_len: int, tkids: List[int]) -> List[int]:
    """Pad token id list when token id list is shorter than required.

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

    See Also
    --------
    lmp.tknzr.BaseTknzr.trunc_to_max
      Truncate token id list when token id list is longer than allowed.

    Examples
    --------
    >>> from lmp.tknzr._base import PAD_TKID
    >>> from lmp.tknzr import CharTknzr
    >>> tknzr = CharTknzr(is_uncased=True, max_vocab=10, min_count=1)
    >>> assert tknzr.pad_to_max(max_seq_len=5, tkids=[1, 2, 3]) == [1, 2, 3, PAD_TKID, PAD_TKID]
    """
    # `max_seq_len` validation.
    lmp.util.validate.raise_if_not_instance(val=max_seq_len, val_name='max_seq_len', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, max_seq_len], val_names=['1', 'max_seq_len'])

    # Calculate padding length.
    pad_len = max(0, max_seq_len - len(tkids))

    # Pad to maximum sequence length.
    return tkids + [PAD_TKID] * pad_len

  def enc(self, max_seq_len: int, txt: str) -> List[int]:
    """Encode text into token id list.

    Text will be tokenized into token list (``tk_0, tk_1, tk_2, ..., tk_n``) and formatted as follow::

      [bos] tk_0 tk_1 [unk] tk_3 ... tk_n [eos] [pad] ... [pad]

    - ``[bos]`` is the "begin of sentence" token.
    - ``[eos]`` is the "end of sentence" token.
    - ``[unk]`` tokens are used to replace :term:`OOV` tokens, i.e., tokens not in tokenizer's dictionary.
      These tokens cannot be encoded since they are, well, unknown.
    - After prepending ``[bos]`` and appending ``[eos]`` tokens, if token list is longer than ``max_seq_len``, then
      token list will be truncated to have length equals to ``max_seq_len``.
    - After prepending ``[bos]`` and appending ``[eos]`` tokens, if token list is shorter than ``max_seq_len``, then
      padding token ``[pad]`` will be appended to token list util token list has length equals to ``max_seq_len``.
    - All tokens in token list are converted to token ids and returned.

    Parameters
    ----------
    max_seq_len: int
      Maximum length of token id list.
    txt: str
      Text to be encoded.

    Returns
    -------
    list[int]
      Encoded token ids list.

    See Also
    --------
    lmp.tknzr.BaseTknzr.dec
      Decode token id list back to text.
    lmp.tknzr.BaseTknzr.pad_to_max
      Pad token id list when token id list is shorter than required.
    lmp.tknzr.BaseTknzr.tknz
      Perform tokenization on text.
    lmp.tknzr.BaseTknzr.trunc_to_max
      Truncate token id list when token id list is longer than allowed.
    """
    # Prepend `[bos]` token id.
    tkids = [BOS_TKID]

    # Convert tokens into token ids.
    for tk in self.tknz(txt):
      # Perform token id lookup.
      try:
        tkids.append(self.tk2id[tk])
      # Convert unknown tokens into `[unk]` token id.
      except KeyError:
        tkids.append(UNK_TKID)

    # Append `[eos]` token id.
    tkids.append(EOS_TKID)

    # First truncate sequence to maximum sequence length, then pad sequence to maximum sequence length.
    return self.pad_to_max(max_seq_len=max_seq_len, tkids=self.trunc_to_max(max_seq_len=max_seq_len, tkids=tkids))

  def dec(self, tkids: List[int], *, rm_sp_tks: bool = False) -> str:
    """Decode token id list back to text.

    :term:`Token id` list will first be converted into token list, then be detokenized back to text.
    Special tokens other than ``[unk]`` will be removed if ``rm_sp_tks == True``.
    Note that unknown tokens ``[unk]`` will not be removed even if ``rm_sp_tks == True``.
    Token ids in the list that are not in tokenizer's inverse lookup table will be converted into ``[unk]`` token.

    Parameters
    ----------
    tkids: list[int]
      Token id list to be decoded.
    rm_sp_tks: bool, default: False
      Set to ``True`` to remove ``[bos]``, ``[eos]`` and ``[pad]``.

    Returns
    -------
    str
      Decoded text.

    See Also
    --------
    lmp.tknzr.BaseTknzr.dtknz
      Convert tokens back to text.
    lmp.tknzr.BaseTknzr.enc
      Encode text into token id list.

    Note
    ----
    Unknown tokens cannot be converted back to original tokens, so unknown tokens should not be removed and serve as
    hints of :term:`OOV`.
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
      # Convert unknown token ids into `[unk]` token.
      except KeyError:
        tks.append(UNK_TK)

    return self.dtknz(tks)

  def batch_enc(self, batch_txt: List[str], max_seq_len: int) -> List[List[int]]:
    """Encode batch of text into batch of token id lists.

    Each text in ``batch_txt`` will be encoded with :py:meth:`lmp.tknzr.BaseTknzr.enc`.
    All encoded token id lists will have the same length (``= max_seq_len``).

    Parameters
    ----------
    batch_txt: list[str],
      Batch of text to be encoded.
    max_seq_len: int
      Maximum length of all token id lists in the batch.

    Returns
    -------
    list[list[int]]
      Encoded batch of token id lists.

    See Also
    --------
    lmp.tknzr.BaseTknzr.pad_to_max
      Pad token id list when token id list is shorter than required.
    lmp.tknzr.BaseTknzr.trunc_to_max
      Truncate token id list when token id list is longer than allowed.
    lmp.tknzr.BaseTknzr.batch_dec
      Decode batch of token id lists back to batch of text.
    lmp.tknzr.BaseTknzr.enc
      Encode text into token id list.
    """
    # `batch_txt` validation.
    lmp.util.validate.raise_if_not_instance(val=batch_txt, val_name='batch_txt', val_type=list)
    return [self.enc(max_seq_len=max_seq_len, txt=txt) for txt in batch_txt]

  def batch_dec(self, batch_tkids: List[List[int]], *, rm_sp_tks: bool = False) -> List[str]:
    """Decode batch of token id lists back to batch of text.

    Parameters
    ----------
    batch_tkids: list[list[int]]
      Batch of token id lists to be decoded.
    rm_sp_tks: bool, default: False
      Whether to remove special tokens.
      See :py:meth:`lmp.tknzr.BaseTknzr.dec` for ``rm_sp_tks`` usage.

    Returns
    -------
    list[str]
      Batch of decoded text.

    See Also
    --------
    lmp.tknzr.BaseTknzr.batch_enc
      Encode batch of text into batch of token id lists.
    lmp.tknzr.BaseTknzr.dec
      Decode token id list back to text.
    """
    # `batch_tkids` validation.
    lmp.util.validate.raise_if_not_instance(val=batch_tkids, val_name='batch_tkids', val_type=list)

    # Decode each sequence of token ids in the batch.
    return [self.dec(tkids, rm_sp_tks=rm_sp_tks) for tkids in batch_tkids]
