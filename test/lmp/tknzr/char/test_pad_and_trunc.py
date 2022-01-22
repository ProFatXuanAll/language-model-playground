"""Test sequence padding and truncation.

Test target:
- :py:meth:`lmp.tknzr.CharTknzr.pad_to_max`.
- :py:meth:`lmp.tknzr.CharTknzr.trunc_to_max`.
"""

from lmp.tknzr import CharTknzr


def test_default() -> None:
  """Don't do anything when `max_seq_len == -1`."""
  assert CharTknzr.pad_to_max([]) == []
  assert CharTknzr.pad_to_max([]) == []
  assert CharTknzr.pad_to_max([
    CharTknzr.bos_tkid,
    CharTknzr.unk_tkid,
    CharTknzr.eos_tkid,
  ]) == [
    CharTknzr.bos_tkid,
    CharTknzr.unk_tkid,
    CharTknzr.eos_tkid,
  ]
  assert CharTknzr.trunc_to_max([]) == []
  assert CharTknzr.trunc_to_max([]) == []
  assert CharTknzr.trunc_to_max([
    CharTknzr.bos_tkid,
    CharTknzr.unk_tkid,
    CharTknzr.eos_tkid,
  ]) == [
    CharTknzr.bos_tkid,
    CharTknzr.unk_tkid,
    CharTknzr.eos_tkid,
  ]


def test_padding() -> None:
  """Pad to specified length."""
  assert CharTknzr.pad_to_max([], max_seq_len=2) == [CharTknzr.pad_tkid, CharTknzr.pad_tkid]
  assert CharTknzr.pad_to_max(
    [
      CharTknzr.bos_tkid,
      CharTknzr.unk_tkid,
      CharTknzr.eos_tkid,
    ],
    max_seq_len=5,
  ) == [
    CharTknzr.bos_tkid,
    CharTknzr.unk_tkid,
    CharTknzr.eos_tkid,
    CharTknzr.pad_tkid,
    CharTknzr.pad_tkid,
  ]


def test_truncate() -> None:
  """Truncate to specified length."""
  assert CharTknzr.trunc_to_max([], max_seq_len=5) == []
  assert CharTknzr.trunc_to_max(
    [
      CharTknzr.bos_tkid,
      CharTknzr.unk_tkid,
      CharTknzr.eos_tkid,
      CharTknzr.pad_tkid,
      CharTknzr.pad_tkid,
    ],
    max_seq_len=2,
  ) == [
    CharTknzr.bos_tkid,
    CharTknzr.unk_tkid,
  ]
