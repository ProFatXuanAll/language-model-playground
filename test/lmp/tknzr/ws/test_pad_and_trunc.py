"""Test sequence padding and truncation.

Test target:
- :py:meth:`lmp.tknzr.WsTknzr.pad_to_max`.
- :py:meth:`lmp.tknzr.WsTknzr.trunc_to_max`.
"""

from lmp.tknzr import WsTknzr


def test_default() -> None:
  """Don't do anything when `max_seq_len == -1`."""
  assert WsTknzr.pad_to_max([], WsTknzr.pad_tk) == []
  assert WsTknzr.pad_to_max([], WsTknzr.pad_tkid) == []
  assert WsTknzr.pad_to_max(
    [
      WsTknzr.bos_tk,
      WsTknzr.unk_tk,
      WsTknzr.eos_tk,
    ],
    WsTknzr.pad_tk,
  ) == [
    WsTknzr.bos_tk,
    WsTknzr.unk_tk,
    WsTknzr.eos_tk,
  ]
  assert WsTknzr.pad_to_max(
    [
      WsTknzr.bos_tkid,
      WsTknzr.unk_tkid,
      WsTknzr.eos_tkid,
    ],
    WsTknzr.pad_tkid,
  ) == [
    WsTknzr.bos_tkid,
    WsTknzr.unk_tkid,
    WsTknzr.eos_tkid,
  ]
  assert WsTknzr.trunc_to_max([]) == []
  assert WsTknzr.trunc_to_max([]) == []
  assert WsTknzr.trunc_to_max([
    WsTknzr.bos_tk,
    WsTknzr.unk_tk,
    WsTknzr.eos_tk,
  ]) == [
    WsTknzr.bos_tk,
    WsTknzr.unk_tk,
    WsTknzr.eos_tk,
  ]
  assert WsTknzr.trunc_to_max([
    WsTknzr.bos_tkid,
    WsTknzr.unk_tkid,
    WsTknzr.eos_tkid,
  ]) == [
    WsTknzr.bos_tkid,
    WsTknzr.unk_tkid,
    WsTknzr.eos_tkid,
  ]


def test_padding() -> None:
  """Pad to specified length."""
  assert WsTknzr.pad_to_max([], WsTknzr.pad_tk, max_seq_len=2) == [WsTknzr.pad_tk, WsTknzr.pad_tk]
  assert WsTknzr.pad_to_max([], WsTknzr.pad_tkid, max_seq_len=2) == [WsTknzr.pad_tkid, WsTknzr.pad_tkid]
  assert WsTknzr.pad_to_max(
    [
      WsTknzr.bos_tk,
      WsTknzr.unk_tk,
      WsTknzr.eos_tk,
    ],
    WsTknzr.pad_tk,
    max_seq_len=5,
  ) == [
    WsTknzr.bos_tk,
    WsTknzr.unk_tk,
    WsTknzr.eos_tk,
    WsTknzr.pad_tk,
    WsTknzr.pad_tk,
  ]
  assert WsTknzr.pad_to_max(
    [
      WsTknzr.bos_tkid,
      WsTknzr.unk_tkid,
      WsTknzr.eos_tkid,
    ],
    WsTknzr.pad_tkid,
    max_seq_len=5,
  ) == [
    WsTknzr.bos_tkid,
    WsTknzr.unk_tkid,
    WsTknzr.eos_tkid,
    WsTknzr.pad_tkid,
    WsTknzr.pad_tkid,
  ]


def test_truncate() -> None:
  """Truncate to specified length."""
  assert WsTknzr.trunc_to_max([], max_seq_len=5) == []
  assert WsTknzr.trunc_to_max([], max_seq_len=5) == []
  assert WsTknzr.trunc_to_max(
    [
      WsTknzr.bos_tk,
      WsTknzr.unk_tk,
      WsTknzr.eos_tk,
      WsTknzr.pad_tk,
      WsTknzr.pad_tk,
    ],
    max_seq_len=2,
  ) == [
    WsTknzr.bos_tk,
    WsTknzr.unk_tk,
  ]
  assert WsTknzr.trunc_to_max(
    [
      WsTknzr.bos_tkid,
      WsTknzr.unk_tkid,
      WsTknzr.eos_tkid,
      WsTknzr.pad_tkid,
      WsTknzr.pad_tkid,
    ],
    max_seq_len=2,
  ) == [
    WsTknzr.bos_tkid,
    WsTknzr.unk_tkid,
  ]
