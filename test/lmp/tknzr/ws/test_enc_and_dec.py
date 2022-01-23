"""Test token encoding and decoding.

Test target:
- :py:meth:`lmp.tknzr.WsTknzr.dec`.
- :py:meth:`lmp.tknzr.WsTknzr.enc`.
"""

from lmp.tknzr import WsTknzr


def test_cased_enc() -> None:
  """Encode text to token ids (case-sensitive)."""
  tk2id = {
    WsTknzr.bos_tk: WsTknzr.bos_tkid,
    WsTknzr.eos_tk: WsTknzr.eos_tkid,
    WsTknzr.pad_tk: WsTknzr.pad_tkid,
    WsTknzr.unk_tk: WsTknzr.unk_tkid,
    'A': max(WsTknzr.bos_tkid, WsTknzr.eos_tkid, WsTknzr.pad_tkid, WsTknzr.unk_tkid) + 1,
    'a': max(WsTknzr.bos_tkid, WsTknzr.eos_tkid, WsTknzr.pad_tkid, WsTknzr.unk_tkid) + 2,
  }
  tknzr = WsTknzr(
    is_uncased=False,
    max_vocab=-1,
    min_count=0,
    tk2id=tk2id,
  )
  # Return `[bos] [eos]` when given empty input.
  assert tknzr.enc(max_seq_len=2, txt='') == [
    WsTknzr.bos_tkid,
    WsTknzr.eos_tkid,
  ]
  # Encoding format.
  assert tknzr.enc(max_seq_len=4, txt='a A') == [
    WsTknzr.bos_tkid,
    tk2id['a'],
    tk2id['A'],
    WsTknzr.eos_tkid,
  ]
  # Padding.
  assert tknzr.enc(max_seq_len=5, txt='a A') == [
    WsTknzr.bos_tkid,
    tk2id['a'],
    tk2id['A'],
    WsTknzr.eos_tkid,
    WsTknzr.pad_tkid,
  ]
  # Truncate.
  assert tknzr.enc(max_seq_len=3, txt='a A') == [
    WsTknzr.bos_tkid,
    tk2id['a'],
    tk2id['A'],
  ]
  # Unknown tokens.
  assert tknzr.enc(max_seq_len=4, txt='b B') == [
    WsTknzr.bos_tkid,
    WsTknzr.unk_tkid,
    WsTknzr.unk_tkid,
    WsTknzr.eos_tkid,
  ]
  # Unknown tokens with padding.
  assert tknzr.enc(max_seq_len=5, txt='b B') == [
    WsTknzr.bos_tkid,
    WsTknzr.unk_tkid,
    WsTknzr.unk_tkid,
    WsTknzr.eos_tkid,
    WsTknzr.pad_tkid,
  ]
  # Unknown tokens with truncation.
  assert tknzr.enc(max_seq_len=2, txt='b B') == [
    WsTknzr.bos_tkid,
    WsTknzr.unk_tkid,
  ]


def test_uncased_enc() -> None:
  """Encode text to token ids (case-insensitive)."""
  tk2id = {
    WsTknzr.bos_tk: WsTknzr.bos_tkid,
    WsTknzr.eos_tk: WsTknzr.eos_tkid,
    WsTknzr.pad_tk: WsTknzr.pad_tkid,
    WsTknzr.unk_tk: WsTknzr.unk_tkid,
    'a': max(WsTknzr.bos_tkid, WsTknzr.eos_tkid, WsTknzr.pad_tkid, WsTknzr.unk_tkid) + 1,
  }
  tknzr = WsTknzr(
    is_uncased=True,
    max_vocab=-1,
    min_count=0,
    tk2id=tk2id,
  )
  # Return `[bos] [eos]` when given empty input.
  assert tknzr.enc(max_seq_len=2, txt='') == [
    WsTknzr.bos_tkid,
    WsTknzr.eos_tkid,
  ]
  # Encoding format.
  assert tknzr.enc(max_seq_len=4, txt='a A') == [
    WsTknzr.bos_tkid,
    tk2id['a'],
    tk2id['a'],
    WsTknzr.eos_tkid,
  ]
  # Padding.
  assert tknzr.enc(max_seq_len=5, txt='a A') == [
    WsTknzr.bos_tkid,
    tk2id['a'],
    tk2id['a'],
    WsTknzr.eos_tkid,
    WsTknzr.pad_tkid,
  ]
  # Truncate.
  assert tknzr.enc(max_seq_len=3, txt='a A') == [
    WsTknzr.bos_tkid,
    tk2id['a'],
    tk2id['a'],
  ]
  # Unknown tokens.
  assert tknzr.enc(max_seq_len=4, txt='b B') == [
    WsTknzr.bos_tkid,
    WsTknzr.unk_tkid,
    WsTknzr.unk_tkid,
    WsTknzr.eos_tkid,
  ]
  # Unknown tokens with padding.
  assert tknzr.enc(max_seq_len=5, txt='b B') == [
    WsTknzr.bos_tkid,
    WsTknzr.unk_tkid,
    WsTknzr.unk_tkid,
    WsTknzr.eos_tkid,
    WsTknzr.pad_tkid,
  ]
  # Unknown tokens with truncation.
  assert tknzr.enc(max_seq_len=2, txt='b B') == [
    WsTknzr.bos_tkid,
    WsTknzr.unk_tkid,
  ]


def test_dec() -> None:
  """Decode token ids to text."""
  tk2id = {
    WsTknzr.bos_tk: WsTknzr.bos_tkid,
    WsTknzr.eos_tk: WsTknzr.eos_tkid,
    WsTknzr.pad_tk: WsTknzr.pad_tkid,
    WsTknzr.unk_tk: WsTknzr.unk_tkid,
    'A': max(WsTknzr.bos_tkid, WsTknzr.eos_tkid, WsTknzr.pad_tkid, WsTknzr.unk_tkid) + 1,
    'a': max(WsTknzr.bos_tkid, WsTknzr.eos_tkid, WsTknzr.pad_tkid, WsTknzr.unk_tkid) + 2,
  }
  tknzr = WsTknzr(
    is_uncased=False,
    max_vocab=-1,
    min_count=0,
    tk2id=tk2id,
  )
  # Return empty string when given empty list.
  assert tknzr.dec(tkids=[]) == ''
  # Decoding format.
  assert tknzr.dec(
    tkids=[
      WsTknzr.bos_tkid,
      tk2id['a'],
      WsTknzr.unk_tkid,
      tk2id['A'],
      WsTknzr.eos_tkid,
      WsTknzr.pad_tkid,
    ],
    rm_sp_tks=False,
  ) == f'{WsTknzr.bos_tk} a {WsTknzr.unk_tk} A {WsTknzr.eos_tk} {WsTknzr.pad_tk}'
  # Remove special tokens but not unknown tokens.
  assert tknzr.dec(
    tkids=[
      WsTknzr.bos_tkid,
      tk2id['a'],
      WsTknzr.unk_tkid,
      tk2id['A'],
      WsTknzr.unk_tkid,
      WsTknzr.eos_tkid,
      WsTknzr.pad_tkid,
    ],
    rm_sp_tks=True,
  ) == f'a {WsTknzr.unk_tk} A {WsTknzr.unk_tk}'
  # Convert unknown id to unknown tokens.
  assert tknzr.dec(
    tkids=[
      WsTknzr.bos_tkid,
      max(tk2id.values()) + 1,
      max(tk2id.values()) + 2,
      WsTknzr.eos_tkid,
      WsTknzr.pad_tkid,
    ],
    rm_sp_tks=False,
  ) == f'{WsTknzr.bos_tk} {WsTknzr.unk_tk} {WsTknzr.unk_tk} {WsTknzr.eos_tk} {WsTknzr.pad_tk}'
