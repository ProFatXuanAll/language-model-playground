"""Test the construction of :py:class:`lmp.tknzr._ws.WsTknzr`.

Test target:
- :py:meth:`lmp.tknzr._ws.WsTknzr.__init__`.
"""

from lmp.tknzr._ws import WsTknzr
from lmp.vars import BOS_TK, BOS_TKID, EOS_TK, EOS_TKID, PAD_TK, PAD_TKID, UNK_TK, UNK_TKID


def test_default_values() -> None:
  """Ensure default values' consistency."""
  tknzr = WsTknzr()
  assert not tknzr.is_uncased
  assert tknzr.max_vocab == -1
  assert tknzr.min_count == 0
  assert tknzr.tk2id == {BOS_TK: BOS_TKID, EOS_TK: EOS_TKID, PAD_TK: PAD_TKID, UNK_TK: UNK_TKID}
  assert tknzr.id2tk == {BOS_TKID: BOS_TK, EOS_TKID: EOS_TK, PAD_TKID: PAD_TK, UNK_TKID: UNK_TK}


def test_good_values(is_uncased: bool, max_vocab: int, min_count: int) -> None:
  """Must correctly construct tokenizer."""
  tknzr = WsTknzr(is_uncased=is_uncased, max_vocab=max_vocab, min_count=min_count)
  assert tknzr.is_uncased == is_uncased
  assert tknzr.max_vocab == max_vocab
  assert tknzr.min_count == min_count
  assert tknzr.tk2id == {BOS_TK: BOS_TKID, EOS_TK: EOS_TKID, PAD_TK: PAD_TKID, UNK_TK: UNK_TKID}
  assert tknzr.id2tk == {BOS_TKID: BOS_TK, EOS_TKID: EOS_TK, PAD_TKID: PAD_TK, UNK_TKID: UNK_TK}
