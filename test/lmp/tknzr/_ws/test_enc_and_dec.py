"""Test token encoding and decoding.

Test target:
- :py:meth:`lmp.tknzr._ws.WsTknzr.dec`.
- :py:meth:`lmp.tknzr._ws.WsTknzr.enc`.
"""

from lmp.tknzr._ws import WsTknzr
from lmp.vars import BOS_TK, BOS_TKID, EOS_TK, EOS_TKID, PAD_TK, PAD_TKID, UNK_TK, UNK_TKID


def test_cased_enc() -> None:
  """Encode text to token ids (case-sensitive)."""
  tknzr = WsTknzr(is_uncased=False, max_vocab=-1, min_count=0)
  tknzr.build_vocab(['A', 'a'])

  # Return `<bos> <eos>` when given empty input.
  assert tknzr.enc(txt='') == [BOS_TKID, EOS_TKID]

  # Encoding format.
  assert tknzr.enc(txt='a A') == [BOS_TKID, tknzr.tk2id['a'], tknzr.tk2id['A'], EOS_TKID]

  # Unknown tokens.
  assert tknzr.enc(txt='b B') == [BOS_TKID, UNK_TKID, UNK_TKID, EOS_TKID]


def test_uncased_enc() -> None:
  """Encode text to token ids (case-insensitive)."""
  tknzr = WsTknzr(is_uncased=True, max_vocab=-1, min_count=0)
  tknzr.build_vocab(batch_txt=['a'])

  # Return `<bos> <eos>` when given empty input.
  assert tknzr.enc(txt='') == [BOS_TKID, EOS_TKID]

  # Encoding format.
  assert tknzr.enc(txt='a A') == [BOS_TKID, tknzr.tk2id['a'], tknzr.tk2id['a'], EOS_TKID]

  # Unknown tokens.
  assert tknzr.enc(txt='b B') == [BOS_TKID, UNK_TKID, UNK_TKID, EOS_TKID]


def test_dec() -> None:
  """Decode token ids to text."""
  tknzr = WsTknzr(is_uncased=False, max_vocab=-1, min_count=0)
  tknzr.build_vocab(batch_txt=['A', 'a'])

  # Return empty string when given empty list.
  assert tknzr.dec(tkids=[]) == ''

  # Decoding format.
  assert tknzr.dec(
    tkids=[
      BOS_TKID,
      tknzr.tk2id['a'],
      UNK_TKID,
      tknzr.tk2id['A'],
      EOS_TKID,
      PAD_TKID,
    ],
    rm_sp_tks=False,
  ) == f'{BOS_TK} a {UNK_TK} A {EOS_TK} {PAD_TK}'

  # Remove special tokens but not unknown tokens.
  assert tknzr.dec(
    tkids=[
      BOS_TKID,
      tknzr.tk2id['a'],
      UNK_TKID,
      tknzr.tk2id['A'],
      UNK_TKID,
      EOS_TKID,
      PAD_TKID,
    ],
    rm_sp_tks=True,
  ) == f'a {UNK_TK} A {UNK_TK}'

  # Convert unknown id to unknown tokens.
  assert tknzr.dec(
    tkids=[
      BOS_TKID,
      max(tknzr.tk2id.values()) + 1,
      max(tknzr.tk2id.values()) + 2,
      EOS_TKID,
      PAD_TKID,
    ],
    rm_sp_tks=False,
  ) == f'{BOS_TK} {UNK_TK} {UNK_TK} {EOS_TK} {PAD_TK}'
