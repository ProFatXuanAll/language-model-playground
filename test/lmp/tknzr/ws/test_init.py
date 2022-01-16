"""Test the construction of :py:mod:`lmp.tknzr.WsTknzr`.

Test target:
- :py:meth:`lmp.tknzr.WsTknzr.__init__`.
"""

from typing import Dict

from lmp.tknzr import WsTknzr


def test_default_values() -> None:
  """Must correctly construct tokenizer using default values."""
  is_uncased = False
  max_vocab = -1
  min_count = 0
  tknzr = WsTknzr(is_uncased=is_uncased, max_vocab=max_vocab, min_count=min_count)
  assert tknzr.is_uncased == is_uncased
  assert tknzr.max_vocab == max_vocab
  assert tknzr.min_count == min_count
  assert tknzr.tk2id == {
    WsTknzr.bos_tk: WsTknzr.bos_tkid,
    WsTknzr.eos_tk: WsTknzr.eos_tkid,
    WsTknzr.pad_tk: WsTknzr.pad_tkid,
    WsTknzr.unk_tk: WsTknzr.unk_tkid,
  }
  assert tknzr.id2tk == {
    WsTknzr.bos_tkid: WsTknzr.bos_tk,
    WsTknzr.eos_tkid: WsTknzr.eos_tk,
    WsTknzr.pad_tkid: WsTknzr.pad_tk,
    WsTknzr.unk_tkid: WsTknzr.unk_tk,
  }


def test_good_values(is_uncased: bool, max_vocab: int, min_count: int, tk2id: Dict[str, int]) -> None:
  """Must correctly construct tokenizer."""
  tknzr = WsTknzr(is_uncased=is_uncased, max_vocab=max_vocab, min_count=min_count, tk2id=tk2id)
  assert tknzr.is_uncased == is_uncased
  assert tknzr.max_vocab == max_vocab
  assert tknzr.min_count == min_count
  assert tknzr.tk2id == tk2id
  assert tknzr.id2tk == {v: k for k, v in tk2id.items()}
