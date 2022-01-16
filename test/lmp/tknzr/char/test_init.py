"""Test the construction of :py:mod:`lmp.tknzr.CharTknzr`.

Test target:
- :py:meth:`lmp.tknzr.CharTknzr.__init__`.
"""

from typing import Dict

from lmp.tknzr import CharTknzr


def test_default_values() -> None:
  """Must correctly construct tokenizer using default values."""
  is_uncased = False
  max_vocab = -1
  min_count = 0
  tknzr = CharTknzr(is_uncased=is_uncased, max_vocab=max_vocab, min_count=min_count)
  assert tknzr.is_uncased == is_uncased
  assert tknzr.max_vocab == max_vocab
  assert tknzr.min_count == min_count
  assert tknzr.tk2id == {
    CharTknzr.bos_tk: CharTknzr.bos_tkid,
    CharTknzr.eos_tk: CharTknzr.eos_tkid,
    CharTknzr.pad_tk: CharTknzr.pad_tkid,
    CharTknzr.unk_tk: CharTknzr.unk_tkid,
  }
  assert tknzr.id2tk == {
    CharTknzr.bos_tkid: CharTknzr.bos_tk,
    CharTknzr.eos_tkid: CharTknzr.eos_tk,
    CharTknzr.pad_tkid: CharTknzr.pad_tk,
    CharTknzr.unk_tkid: CharTknzr.unk_tk,
  }


def test_good_values(is_uncased: bool, max_vocab: int, min_count: int, tk2id: Dict[str, int]) -> None:
  """Must correctly construct tokenizer."""
  tknzr = CharTknzr(is_uncased=is_uncased, max_vocab=max_vocab, min_count=min_count, tk2id=tk2id)
  assert tknzr.is_uncased == is_uncased
  assert tknzr.max_vocab == max_vocab
  assert tknzr.min_count == min_count
  assert tknzr.tk2id == tk2id
  assert tknzr.id2tk == {v: k for k, v in tk2id.items()}
