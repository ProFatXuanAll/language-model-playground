"""Test sequence padding and truncation.

Test target:
- :py:meth:`lmp.tknzr.WsTknzr.pad_to_max`.
- :py:meth:`lmp.tknzr.WsTknzr.trunc_to_max`.
"""

from lmp.tknzr import WsTknzr


def test_padding() -> None:
  """Pad to specified length."""
  assert WsTknzr.pad_to_max(max_seq_len=2, tkids=[]) == [WsTknzr.pad_tkid, WsTknzr.pad_tkid]
  assert WsTknzr.pad_to_max(
    max_seq_len=5,
    tkids=[
      WsTknzr.bos_tkid,
      WsTknzr.unk_tkid,
      WsTknzr.eos_tkid,
    ],
  ) == [
    WsTknzr.bos_tkid,
    WsTknzr.unk_tkid,
    WsTknzr.eos_tkid,
    WsTknzr.pad_tkid,
    WsTknzr.pad_tkid,
  ]


def test_truncate() -> None:
  """Truncate to specified length."""
  assert WsTknzr.trunc_to_max(max_seq_len=5, tkids=[]) == []
  assert WsTknzr.trunc_to_max(
    max_seq_len=2,
    tkids=[
      WsTknzr.bos_tkid,
      WsTknzr.unk_tkid,
      WsTknzr.eos_tkid,
      WsTknzr.pad_tkid,
      WsTknzr.pad_tkid,
    ],
  ) == [
    WsTknzr.bos_tkid,
    WsTknzr.unk_tkid,
  ]
