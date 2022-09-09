"""Test sequence padding and truncation.

Test target:
- :py:meth:`lmp.tknzr._bpe.BPETknzr.pad_to_max`.
- :py:meth:`lmp.tknzr._bpe.BPETknzr.trunc_to_max`.
"""

from lmp.tknzr._bpe import BPETknzr
from lmp.vars import BOS_TKID, EOS_TKID, PAD_TKID, UNK_TKID


def test_padding(is_uncased: bool, max_vocab: int, min_count: int) -> None:
  """Pad to specified length."""
  assert BPETknzr(
    is_uncased=is_uncased,
    max_vocab=max_vocab,
    min_count=min_count,
  ).pad_to_max(max_seq_len=2, tkids=[]) == [
    PAD_TKID,
    PAD_TKID,
  ]
  assert BPETknzr(
    is_uncased=is_uncased,
    max_vocab=max_vocab,
    min_count=min_count,
  ).pad_to_max(max_seq_len=5, tkids=[
    BOS_TKID,
    UNK_TKID,
    EOS_TKID,
  ]) == [
    BOS_TKID,
    UNK_TKID,
    EOS_TKID,
    PAD_TKID,
    PAD_TKID,
  ]
