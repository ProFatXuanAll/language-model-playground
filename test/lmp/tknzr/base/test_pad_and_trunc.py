"""Test sequence padding and truncation.

Test target:
- :py:meth:`lmp.tknzr.BaseTknzr.pad_to_max`.
- :py:meth:`lmp.tknzr.BaseTknzr.trunc_to_max`.
"""

from lmp.tknzr import BaseTknzr


def test_padding() -> None:
  """Pad to specified length."""
  assert BaseTknzr.pad_to_max(max_seq_len=2, tkids=[]) == [BaseTknzr.pad_tkid, BaseTknzr.pad_tkid]
  assert BaseTknzr.pad_to_max(
    max_seq_len=5,
    tkids=[
      BaseTknzr.bos_tkid,
      BaseTknzr.unk_tkid,
      BaseTknzr.eos_tkid,
    ],
  ) == [
    BaseTknzr.bos_tkid,
    BaseTknzr.unk_tkid,
    BaseTknzr.eos_tkid,
    BaseTknzr.pad_tkid,
    BaseTknzr.pad_tkid,
  ]


def test_truncate() -> None:
  """Truncate to specified length."""
  assert BaseTknzr.trunc_to_max(max_seq_len=5, tkids=[]) == []
  assert BaseTknzr.trunc_to_max(
    max_seq_len=2,
    tkids=[
      BaseTknzr.bos_tkid,
      BaseTknzr.unk_tkid,
      BaseTknzr.eos_tkid,
      BaseTknzr.pad_tkid,
      BaseTknzr.pad_tkid,
    ],
  ) == [
    BaseTknzr.bos_tkid,
    BaseTknzr.unk_tkid,
  ]
