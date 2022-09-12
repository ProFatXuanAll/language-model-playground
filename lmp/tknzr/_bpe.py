"""Byte-Paired Encoding tokenizer class.

Attributes
----------
EOW: typing.Final[str]
  Special token indicating end-of-word.
SPLIT_PTTN: typing.Final[re.Pattern]
  Special tokens and whitespaces matching pattern.
"""

import argparse
import copy
import re
import typing
from collections import Counter
from typing import Any, ClassVar, Dict, Final, Iterable, List, Tuple

# Typeshed for `tqdm` is not available, we ignore type check on `tqdm`.
from tqdm import tqdm  # type: ignore

import lmp.util.validate
from lmp.tknzr._base import BaseTknzr
from lmp.vars import SP_TKS

EOW_TK: Final[str] = '<eow>'
SPLIT_PTTN: Final[re.Pattern] = re.compile('(' + '|'.join(map(re.escape, SP_TKS)) + r'|\s+' + ')')


class BPETknzr(BaseTknzr):
  """Byte-Pair Encoding :footcite:`sennrich2016neural` tokenizer class.

  Tokenize text into list of subwords.
  When ``max_vocab`` is set to ``-1``, this tokenizer will contain every unicode character and every whitespace
  separated tokens in its vocabulary.

  Parameters
  ----------
  is_uncased: bool, default: False
    Set to ``True`` to convert text into lowercase.
    Mainly used by :py:meth:`~norm`.
  max_vocab: int, default: -1
    Tokenizer's maximum vocabulary size.
    Set to ``-1`` to include as many token in vocabulary as possible.
    Mainly used by :py:meth:`~build_vocab`.
  min_count: int, default: 0
    Minimum token occurrence counts.
    Subwords have occurrence counts less than ``min_count`` will not be added to tokenizer's vocabulary.
    Mainly used by :py:meth:`~build_vocab`.
  n_merge: int, default: 10000
    Maximum number of merging operation to perform.
  kwargs: typing.Any, optional
    Useless parameter.
    Intently left for subclasses inheritance.

  Attributes
  ----------
  id2tk: dict[int, str]
    Byte-to-id inverse lookup table.
  is_uncased: bool
    Convert text into lowercase if set to ``True``.
  max_vocab: int
    Tokenizer's maximum vocabulary size.
  min_count: int
    Minimum token occurrence counts.
  n_merge: int
    Maximum number of merging operation to perform.
  tk2id: dict[str, int]
    Subword-to-id lookup table.
  tknzr_name: typing.ClassVar[str]
    CLI name of BPE tokenizer is ``BPE``.

  See Also
  --------
  :doc:`lmp.tknzr </tknzr/index>`
    All available tokenizers.

  Examples
  --------
  >>> from lmp.tknzr import BPETknzr
  >>> tknzr = BPETknzr()
  >>> assert tknzr.tknz('abc def') == ['abc<eow>', 'def<eow>']
  >>> assert tknzr.dtknz(['abc<eow>', 'def<eow>']) == 'abc def'
  """

  tknzr_name: ClassVar[str] = 'BPE'

  def __init__(
    self,
    *,
    is_uncased: bool = False,
    max_vocab: int = -1,
    min_count: int = 0,
    n_merge: int = 10000,
    **kwargs: Any,
  ):
    super().__init__(
      is_uncased=is_uncased,
      max_vocab=max_vocab,
      min_count=min_count,
    )

    # `n_merge` validation.
    lmp.util.validate.raise_if_not_instance(val=n_merge, val_name='n_merge', val_type=int)
    lmp.util.validate.raise_if_wrong_ordered(vals=[1, n_merge], val_names=['1', 'n_merge'])
    self.n_merge = n_merge

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
    >>> from lmp.tknzr import BPETknzr
    >>> parser = argparse.ArgumentParser()
    >>> BPETknzr.add_CLI_args(parser)
    >>> args = parser.parse_args([
    ...   '--max_vocab', '10',
    ...   '--min_count', '2',
    ...   '--n_merge', '5000',
    ... ])
    >>> assert args.is_uncased == False
    >>> assert args.max_vocab == 10
    >>> assert args.min_count == 2
    >>> assert args.n_merge == 5000
    """
    super().add_CLI_args(parser=parser)

    # Add BPE tokenizer hyperparameters to CLI arguments.
    group = parser.add_argument_group('BPE tokenizer hyperparameters')
    group.add_argument(
      '--n_merge',
      default=10000,
      help='''
      Number of times to merge token pair.
      Default is ``10000``.
      ''',
      type=int,
    )

  def build_vocab(self, batch_txt: Iterable[str]) -> None:
    """Build tokenizer's vocabulary.

    Build vocabulary based on subword occurrence counts.
    Text in ``batch_txt`` is first normalized and splited into unicode characters.
    All unicode characters having occurrence count higher than ``self.min_count`` are included into vocabulary.
    After adding unicode characters to vocabulary, we treat each unicode character as subword and merge subword pairs
    with cooccurrence count higher than ``self.min_count`` into new subword.
    Merging operation is done at most ``self.n_merge`` times.
    After stopping merging subword, we add subwords with high occurrence count into vocabulary.

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
    ~vocab_size
      Tokenizer's vocabulary size.
    """
    # `batch_txt` validation.
    lmp.util.validate.raise_if_not_instance(val=batch_txt, val_name='batch_txt', val_type=Iterable)

    # Count unique unicode characters.
    char_counter: typing.Counter = Counter()
    char_tuple_db: typing.Counter = Counter()
    for txt in tqdm(batch_txt, desc='Count unicode characters', dynamic_ncols=True):
      # Normalize text and separate text by whitespaces.
      # Special token are treated as an unit and thus are not splited.
      tks = list(filter(bool, [tk.strip() for tk in SPLIT_PTTN.split(self.norm(txt=txt))]))

      # Each whitespace separated token will be splited into a tuple of unicode characters.
      # A EOW (end-of-word) token is appended to the end unicode character in a tuple.
      # For example, The word "low" is transformed into "l o w<eos>".
      for tk in tks:
        # Skip special token.
        if tk in SP_TKS:
          continue

        char_tuple = tuple(tk[:-1]) + (tk[-1] + EOW_TK,)
        char_tuple_db[char_tuple] += 1
        char_counter.update(char_tuple)

    # Include all unicode characters into vocabulary.
    max_id = max(self.tk2id.values()) + 1
    for char_tk, char_tk_freq in tqdm(
      char_counter.most_common(),
      desc='Add unicode characters to vocabulary',
      dynamic_ncols=True,
    ):
      # Stop adding tokens when vocabulary size limit is exceeded.
      # Add as many tokens into vocabulary as possible when `self.max_vocab == -1`.
      if self.max_vocab != -1 and len(self.tk2id) >= self.max_vocab:
        return

      # Stop adding the token when the token occurrence count is low.
      # Since we sort token by occurrence count, tokens in the remaining loops will not have occurrence count higher
      # than `self.min_count` and thus we can break loop safely.
      if char_tk_freq < self.min_count:
        break

      # Skip token if that token is already in vocabulary.
      if char_tk in self.tk2id:
        continue

      # Add new unicode character token to vocabulary.
      self.tk2id[char_tk] = max_id
      self.id2tk[max_id] = char_tk

      # Increment token id.
      max_id += 1

    # Convert to list for O(1) lookup.
    tk_db = list(char_tuple_db.items())

    # Loop through each token pair to count token pair occurrence.
    # Also build index for fast lookup `tk_db` list.
    tk_pair_counter: typing.Counter = Counter()
    tk_pair_index: Dict[Tuple, Dict[int, int]] = {}
    for db_idx, (tk_tuple, tk_tuple_freq) in tqdm(
      enumerate(tk_db),
      desc='Count token pairs and build indices',
      dynamic_ncols=True,
    ):
      for tk_pair in zip(tk_tuple[:-1], tk_tuple[1:]):
        # Accumulate token pair occurence counts.
        tk_pair_counter[tk_pair] += tk_tuple_freq

        if tk_pair not in tk_pair_index:
          tk_pair_index[tk_pair] = {}
        if db_idx not in tk_pair_index[tk_pair]:
          tk_pair_index[tk_pair][db_idx] = 0

        # Token pair can appear in the same character sequence multiple times.
        # For example: in the word "likelihood", token pair "li" appear twice.
        tk_pair_index[tk_pair][db_idx] += 1

    # Merge the most frequent token pair for at most ``self.n_merge`` number of times.
    for _ in tqdm(range(self.n_merge), desc='Merge frequent token pairs', dynamic_ncols=True):
      # Stop merging if all token have been merged.
      if not tk_pair_counter:
        break

      # Find the most frequent token pair and merge it to form a new token.
      most_freq_tk_pair = tk_pair_counter.most_common(n=1)[0][0]
      most_freq_tk_pair_pttn = re.compile(r'\s' + re.escape(' '.join(most_freq_tk_pair)) + r'\s')
      merged_tk = ''.join(most_freq_tk_pair)

      # Stop merging token pair when all token pairs have frequency smaller than ``self.min_count``.
      if tk_pair_counter[most_freq_tk_pair] < self.min_count:
        break

      # Loop through token database to find every occurrence of the most frequent token pair.
      # We do this using the database index `tk_pair_index` we previously built.
      # When an occurrence is found, we replace the occurrence with the newly merged token.
      for db_idx, tk_pair_multiplier in copy.deepcopy(list(tk_pair_index[most_freq_tk_pair].items())):
        tk_tuple, tk_tuple_freq = tk_db[db_idx]

        # The merging token pair and the token appeared right before the merging token pair together form a new token
        # pair; similarly for the token appeared right after the merging token pair.
        # We record them to update token pair occurrence count.
        pre_tk_pairs = []
        post_tk_pairs = []
        for tk_1, tk_2, tk_3 in zip(tk_tuple[:-2], tk_tuple[1:-1], tk_tuple[2:]):
          if (tk_1, tk_2) == most_freq_tk_pair:
            post_tk_pairs.append((tk_2, tk_3))
          if (tk_2, tk_3) == most_freq_tk_pair:
            pre_tk_pairs.append((tk_1, tk_2))

        # Update token pair occurrence count for the token appeared right before the merging token pair.
        for reduced_tk_pair in pre_tk_pairs:
          # Subtract frequency since token pair is merged and no longer exist.
          tk_pair_counter[reduced_tk_pair] -= tk_tuple_freq
          if tk_pair_counter[reduced_tk_pair] <= 0:
            del tk_pair_counter[reduced_tk_pair]

          # Remove index since token pair is merged and no longer exist.
          if reduced_tk_pair in tk_pair_index:
            if db_idx in tk_pair_index[reduced_tk_pair]:
              tk_pair_index[reduced_tk_pair][db_idx] -= 1

              if tk_pair_index[reduced_tk_pair][db_idx] <= 0:
                del tk_pair_index[reduced_tk_pair][db_idx]

            if not tk_pair_index[reduced_tk_pair]:
              del tk_pair_index[reduced_tk_pair]

          # Form a new token pair.
          # We need to add new token pair into `tk_pair_counter` and `tk_pair_index`.
          new_tk_pair = (reduced_tk_pair[0], merged_tk)
          tk_pair_counter[new_tk_pair] += tk_tuple_freq

          if new_tk_pair not in tk_pair_index:
            tk_pair_index[new_tk_pair] = {}
          if db_idx not in tk_pair_index[new_tk_pair]:
            tk_pair_index[new_tk_pair][db_idx] = 0
          tk_pair_index[new_tk_pair][db_idx] += 1

        # Update token pair occurrence count for the token appeared right after the merging token pair.
        for reduced_tk_pair in post_tk_pairs:
          # Subtract frequency since token pair is merged and no longer exist.
          tk_pair_counter[reduced_tk_pair] -= tk_tuple_freq
          if tk_pair_counter[reduced_tk_pair] <= 0:
            del tk_pair_counter[reduced_tk_pair]

          # Remove index since token pair is merged and no longer exist.
          if reduced_tk_pair in tk_pair_index:
            if db_idx in tk_pair_index[reduced_tk_pair]:
              tk_pair_index[reduced_tk_pair][db_idx] -= 1

              if tk_pair_index[reduced_tk_pair][db_idx] <= 0:
                del tk_pair_index[reduced_tk_pair][db_idx]

            if not tk_pair_index[reduced_tk_pair]:
              del tk_pair_index[reduced_tk_pair]

          # Form a new token pair.
          # We need to add new token pair into `tk_pair_counter` and `tk_pair_index`.
          new_tk_pair = (merged_tk, reduced_tk_pair[1])
          tk_pair_counter[new_tk_pair] += tk_tuple_freq

          if new_tk_pair not in tk_pair_index:
            tk_pair_index[new_tk_pair] = {}
          if db_idx not in tk_pair_index[new_tk_pair]:
            tk_pair_index[new_tk_pair][db_idx] = 0
          tk_pair_index[new_tk_pair][db_idx] += 1

        # Replace most frequent token pair in the original tuple with the merged token.
        new_txt = most_freq_tk_pair_pttn.sub(f' {merged_tk} ', f' {" ".join(tk_tuple)} ')
        new_tk_tuple = tuple(new_txt.strip().split(' '))
        tk_db[db_idx] = (new_tk_tuple, tk_tuple_freq)

        # Merged token can form a new token pair by itself.
        if tk_pair_multiplier > 1:
          new_tk_pair = (merged_tk, merged_tk)
          for tk_1, tk_2 in zip(new_tk_tuple[:-1], new_tk_tuple[1:]):
            if tk_1 == tk_2 == merged_tk:
              tk_pair_counter[new_tk_pair] += tk_tuple_freq

              if new_tk_pair not in tk_pair_index:
                tk_pair_index[new_tk_pair] = {}
              if db_idx not in tk_pair_index[new_tk_pair]:
                tk_pair_index[new_tk_pair][db_idx] = 0
              tk_pair_index[new_tk_pair][db_idx] += 1

      # Once a token pair is merged, it no longer need to be consider again.
      del tk_pair_counter[most_freq_tk_pair]

    # Include frequent token into vocabulary.
    tk_counter: typing.Counter = Counter()
    for tk_tuple, tk_tuple_freq in tqdm(tk_db, desc='Count merged tokens', dynamic_ncols=True):
      for tk in tk_tuple:
        tk_counter[tk] += tk_tuple_freq

    max_id = max(self.tk2id.values()) + 1
    for tk, tk_freq in tqdm(tk_counter.most_common(), desc='Add merged tokens to vocabulary', dynamic_ncols=True):
      # Stop adding tokens when vocabulary size limit is exceeded.
      # Add as many tokens into vocabulary as possible when `self.max_vocab == -1`.
      if self.max_vocab != -1 and len(self.tk2id) >= self.max_vocab:
        break

      # Stop adding the token when the token occurrence count is low.
      # Since we sort token by occurrence count, tokens in the remaining loops will not have occurrence count higher
      # than `self.min_count` and thus we can break loop safely.
      if tk_freq < self.min_count:
        break

      # Skip token if that token is already in vocabulary.
      if tk in self.tk2id:
        continue

      # Add new unicode character token to vocabulary.
      self.tk2id[tk] = max_id
      self.id2tk[max_id] = tk

      # Increment token id.
      max_id += 1

  def tknz(self, txt: str) -> List[str]:
    """Convert text into list of words and subwords.

    Text is first normalized then splitted by whitespace.
    Each whitespace separated token is then converted into a word or list of subwords based on whether that token is in
    vocabulary or not.
    Each special token is treated as an unit and thus is not splitted.

    Parameters
    ----------
    txt: str
      Text to be tokenized.

    Returns
    -------
    list[str]
      List of words and subwords.

    See Also
    --------
    ~dtknz
      Convert list of words and subwords back to text.
    ~norm
      Text normalization.

    Examples
    --------
    >>> from lmp.tknzr import BPETknzr
    >>> tknzr = BPETknzr()
    >>> assert tknzr.tknz('abc def') == ['abc<eow>', 'def<eow>']
    """
    # Perform normalization.
    txt = self.norm(txt)

    # Split by whitespaces and special tokens.
    ws_tks = list(filter(bool, [tk.strip() for tk in SPLIT_PTTN.split(txt)]))

    # Add EOW to each whitespace separated token.
    # Special tokens will not have EOW appended to them.
    ws_tks = list(map(lambda tk: tk + EOW_TK if tk not in SP_TKS else tk, ws_tks))

    # Tokenize text.
    tks = []
    for ws_tk in ws_tks:
      # If a token is in vocabulary, treat it as a word.
      if ws_tk in self.tk2id:
        tks.append(ws_tk)
      # Otherwise split that token into subwords.
      else:
        # Longest matching subword in the vocabulary will be retrieved first.
        ws_tk_idx = len(ws_tk) - len(EOW_TK) - 1
        while ws_tk_idx > 0:
          subword = ws_tk[:ws_tk_idx]
          # Find the longest matching subword in the vocabulary.
          if subword in self.tk2id:
            tks.append(subword)
            ws_tk = ws_tk[ws_tk_idx:]
            ws_tk_idx = len(ws_tk) - len(EOW_TK) - 1
          # Do not find matching subword, so reduce matching pattern length.
          else:
            ws_tk_idx -= 1

        # Add the remaining subwords.
        # Note that the remaining subwords may not in vocabulary.
        tks.append(ws_tk[ws_tk_idx:])

    return tks

  def dtknz(self, tks: List[str]) -> str:
    """Convert list of words and subwords back to text.

    First of all, subwords are joined into word without whitespaces and EOS are removed.
    Then words are joined with whitespaces.
    Returned text is normalized.

    Parameters
    ----------
    tks: list[str]
      List of words and subwords to be joint.

    Returns
    -------
    str
      Normalized text with whitespaces in between.

    See Also
    --------
    ~tknz
      Convert text into list of words and subwords.
    ~norm
      Text normalization.

    Examples
    --------
    >>> from lmp.tknzr import BPETknzr
    >>> tknzr = BPETknzr()
    >>> assert tknzr.dtknz(['abc<eow>', 'def<eow>']) == 'abc def'
    """
    txt = ''
    for tk in tks:
      tk = tk.strip()
      txt += tk
      # Seeing a EOW token indicates a subword is ended.
      # We simply insert whitespace after it.
      # Special tokens are treated as words.
      if tk.endswith(EOW_TK) or tk in SP_TKS:
        txt += ' '

    # Remove all occurrences of EOW token.
    txt = re.sub(re.escape(EOW_TK), '', txt)

    # Return text is normalized.
    return self.norm(txt)
