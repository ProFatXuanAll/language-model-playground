"""Test token encoding and decoding.

Test target:
- :py:meth:`lmp.tknzr.CharTknzr.dec`.
- :py:meth:`lmp.tknzr.CharTknzr.enc`.
"""

from lmp.tknzr import CharTknzr


def test_cased_enc() -> None:
  """Encode text to token ids (case-sensitive)."""
  tk2id = {
    CharTknzr.bos_tk: CharTknzr.bos_tkid,
    CharTknzr.eos_tk: CharTknzr.eos_tkid,
    CharTknzr.pad_tk: CharTknzr.pad_tkid,
    CharTknzr.unk_tk: CharTknzr.unk_tkid,
    'A': max(CharTknzr.bos_tkid, CharTknzr.eos_tkid, CharTknzr.pad_tkid, CharTknzr.unk_tkid) + 1,
    'a': max(CharTknzr.bos_tkid, CharTknzr.eos_tkid, CharTknzr.pad_tkid, CharTknzr.unk_tkid) + 2,
  }
  tknzr = CharTknzr(
    is_uncased=False,
    max_vocab=-1,
    min_count=0,
    tk2id=tk2id,
  )
  # Return `[bos] [eos]` when given empty input.
  assert tknzr.enc('') == [
    CharTknzr.bos_tkid,
    CharTknzr.eos_tkid,
  ]
  # Encoding format.
  assert tknzr.enc('aA') == [
    CharTknzr.bos_tkid,
    tk2id['a'],
    tk2id['A'],
    CharTknzr.eos_tkid,
  ]
  # Padding.
  assert tknzr.enc('aA', max_seq_len=5) == [
    CharTknzr.bos_tkid,
    tk2id['a'],
    tk2id['A'],
    CharTknzr.eos_tkid,
    CharTknzr.pad_tkid,
  ]
  # Truncate.
  assert tknzr.enc('aA', max_seq_len=3) == [
    CharTknzr.bos_tkid,
    tk2id['a'],
    tk2id['A'],
  ]
  # Unknown tokens.
  assert tknzr.enc('bB') == [
    CharTknzr.bos_tkid,
    CharTknzr.unk_tkid,
    CharTknzr.unk_tkid,
    CharTknzr.eos_tkid,
  ]
  # Unknown tokens with padding.
  assert tknzr.enc('bB', max_seq_len=5) == [
    CharTknzr.bos_tkid,
    CharTknzr.unk_tkid,
    CharTknzr.unk_tkid,
    CharTknzr.eos_tkid,
    CharTknzr.pad_tkid,
  ]
  # Unknown tokens with truncation.
  assert tknzr.enc('bB', max_seq_len=2) == [
    CharTknzr.bos_tkid,
    CharTknzr.unk_tkid,
  ]


def test_uncased_enc() -> None:
  """Encode text to token ids (case-insensitive)."""
  tk2id = {
    CharTknzr.bos_tk: CharTknzr.bos_tkid,
    CharTknzr.eos_tk: CharTknzr.eos_tkid,
    CharTknzr.pad_tk: CharTknzr.pad_tkid,
    CharTknzr.unk_tk: CharTknzr.unk_tkid,
    'a': max(CharTknzr.bos_tkid, CharTknzr.eos_tkid, CharTknzr.pad_tkid, CharTknzr.unk_tkid) + 1,
  }
  tknzr = CharTknzr(
    is_uncased=True,
    max_vocab=-1,
    min_count=0,
    tk2id=tk2id,
  )
  # Return `[bos] [eos]` when given empty input.
  assert tknzr.enc('') == [
    CharTknzr.bos_tkid,
    CharTknzr.eos_tkid,
  ]
  # Encoding format.
  assert tknzr.enc('aA') == [
    CharTknzr.bos_tkid,
    tk2id['a'],
    tk2id['a'],
    CharTknzr.eos_tkid,
  ]
  # Padding.
  assert tknzr.enc('aA', max_seq_len=5) == [
    CharTknzr.bos_tkid,
    tk2id['a'],
    tk2id['a'],
    CharTknzr.eos_tkid,
    CharTknzr.pad_tkid,
  ]
  # Truncate.
  assert tknzr.enc('aA', max_seq_len=3) == [
    CharTknzr.bos_tkid,
    tk2id['a'],
    tk2id['a'],
  ]
  # Unknown tokens.
  assert tknzr.enc('bB') == [
    CharTknzr.bos_tkid,
    CharTknzr.unk_tkid,
    CharTknzr.unk_tkid,
    CharTknzr.eos_tkid,
  ]
  # Unknown tokens with padding.
  assert tknzr.enc('bB', max_seq_len=5) == [
    CharTknzr.bos_tkid,
    CharTknzr.unk_tkid,
    CharTknzr.unk_tkid,
    CharTknzr.eos_tkid,
    CharTknzr.pad_tkid,
  ]
  # Unknown tokens with truncation.
  assert tknzr.enc('bB', max_seq_len=2) == [
    CharTknzr.bos_tkid,
    CharTknzr.unk_tkid,
  ]


def test_dec() -> None:
  """Decode token ids to text."""
  tk2id = {
    CharTknzr.bos_tk: CharTknzr.bos_tkid,
    CharTknzr.eos_tk: CharTknzr.eos_tkid,
    CharTknzr.pad_tk: CharTknzr.pad_tkid,
    CharTknzr.unk_tk: CharTknzr.unk_tkid,
    'A': max(CharTknzr.bos_tkid, CharTknzr.eos_tkid, CharTknzr.pad_tkid, CharTknzr.unk_tkid) + 1,
    'a': max(CharTknzr.bos_tkid, CharTknzr.eos_tkid, CharTknzr.pad_tkid, CharTknzr.unk_tkid) + 2,
  }
  tknzr = CharTknzr(
    is_uncased=False,
    max_vocab=-1,
    min_count=0,
    tk2id=tk2id,
  )
  # Return empty string when given empty list.
  assert tknzr.dec([]) == ''
  # Decoding format.
  assert tknzr.dec(
    [
      CharTknzr.bos_tkid,
      tk2id['a'],
      CharTknzr.unk_tkid,
      tk2id['A'],
      CharTknzr.eos_tkid,
      CharTknzr.pad_tkid,
    ],
    rm_sp_tks=False,
  ) == f'{CharTknzr.bos_tk}a{CharTknzr.unk_tk}A{CharTknzr.eos_tk}{CharTknzr.pad_tk}'
  # Remove special tokens but not unknown tokens.
  assert tknzr.dec(
    [
      CharTknzr.bos_tkid,
      tk2id['a'],
      CharTknzr.unk_tkid,
      tk2id['A'],
      CharTknzr.unk_tkid,
      CharTknzr.eos_tkid,
      CharTknzr.pad_tkid,
    ],
    rm_sp_tks=True,
  ) == f'a{CharTknzr.unk_tk}A{CharTknzr.unk_tk}'
  # Convert unknown id to unknown tokens.
  assert tknzr.dec(
    [
      CharTknzr.bos_tkid,
      max(tk2id.values()) + 1,
      max(tk2id.values()) + 2,
      CharTknzr.eos_tkid,
      CharTknzr.pad_tkid,
    ],
    rm_sp_tks=False,
  ) == f'{CharTknzr.bos_tk}{CharTknzr.unk_tk}{CharTknzr.unk_tk}{CharTknzr.eos_tk}{CharTknzr.pad_tk}'
