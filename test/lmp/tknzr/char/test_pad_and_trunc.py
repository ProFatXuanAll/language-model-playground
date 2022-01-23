"""Test sequence padding and truncation.

Test target:
- :py:meth:`lmp.tknzr.CharTknzr.pad_to_max`.
- :py:meth:`lmp.tknzr.CharTknzr.trunc_to_max`.
"""

from lmp.tknzr import CharTknzr


def test_padding() -> None:
  """Pad to specified length."""
  assert CharTknzr.pad_to_max(max_seq_len=2, tkids=[]) == [CharTknzr.pad_tkid, CharTknzr.pad_tkid]
  assert CharTknzr.pad_to_max(
    max_seq_len=5,
    tkids=[
      CharTknzr.bos_tkid,
      CharTknzr.unk_tkid,
      CharTknzr.eos_tkid,
    ],
  ) == [
    CharTknzr.bos_tkid,
    CharTknzr.unk_tkid,
    CharTknzr.eos_tkid,
    CharTknzr.pad_tkid,
    CharTknzr.pad_tkid,
  ]


def test_truncate() -> None:
  """Truncate to specified length."""
  assert CharTknzr.trunc_to_max(max_seq_len=5, tkids=[]) == []
  assert CharTknzr.trunc_to_max(
    max_seq_len=2,
    tkids=[
      CharTknzr.bos_tkid,
      CharTknzr.unk_tkid,
      CharTknzr.eos_tkid,
      CharTknzr.pad_tkid,
      CharTknzr.pad_tkid,
    ],
  ) == [
    CharTknzr.bos_tkid,
    CharTknzr.unk_tkid,
  ]
