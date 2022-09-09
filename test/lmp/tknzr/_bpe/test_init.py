"""Test the construction of :py:class:`lmp.tknzr._bpe.BPETknzr`.

Test target:
- :py:meth:`lmp.tknzr._bpe.BPETknzr.__init__`.
"""

from lmp.tknzr._bpe import BPETknzr
from lmp.vars import BOS_TK, BOS_TKID, EOS_TK, EOS_TKID, PAD_TK, PAD_TKID, UNK_TK, UNK_TKID


def test_default_values() -> None:
  """Ensure default values' consistency."""
  tknzr = BPETknzr()
  assert not tknzr.is_uncased
  assert tknzr.max_vocab == -1
  assert tknzr.min_count == 0
  assert tknzr.n_merge == 10000
  assert tknzr.tk2id == {BOS_TK: BOS_TKID, EOS_TK: EOS_TKID, PAD_TK: PAD_TKID, UNK_TK: UNK_TKID}
  assert tknzr.id2tk == {BOS_TKID: BOS_TK, EOS_TKID: EOS_TK, PAD_TKID: PAD_TK, UNK_TKID: UNK_TK}


def test_good_values(is_uncased: bool, max_vocab: int, min_count: int, n_merge: int) -> None:
  """Must correctly construct tokenizer."""
  tknzr = BPETknzr(is_uncased=is_uncased, max_vocab=max_vocab, min_count=min_count, n_merge=n_merge)
  assert tknzr.is_uncased == is_uncased
  assert tknzr.max_vocab == max_vocab
  assert tknzr.min_count == min_count
  assert tknzr.n_merge == n_merge
  assert tknzr.tk2id == {BOS_TK: BOS_TKID, EOS_TK: EOS_TKID, PAD_TK: PAD_TKID, UNK_TK: UNK_TKID}
  assert tknzr.id2tk == {BOS_TKID: BOS_TK, EOS_TKID: EOS_TK, PAD_TKID: PAD_TK, UNK_TKID: UNK_TK}
