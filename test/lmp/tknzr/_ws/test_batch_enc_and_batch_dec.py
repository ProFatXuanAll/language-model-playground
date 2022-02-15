"""Test token encoding and decoding.

Test target:
- :py:meth:`lmp.tknzr._ws.WsTknzr.batch_dec`.
- :py:meth:`lmp.tknzr._ws.WsTknzr.batch_enc`.
"""

from lmp.tknzr._base import BOS_TK, BOS_TKID, EOS_TK, EOS_TKID, PAD_TK, PAD_TKID, UNK_TK, UNK_TKID
from lmp.tknzr._ws import WsTknzr


def test_cased_batch_enc() -> None:
  """Encode batch of text to batch of token ids (case-sensitive)."""
  tknzr = WsTknzr(is_uncased=False, max_seq_len=1, max_vocab=-1, min_count=0)
  tknzr.build_vocab(batch_txt=['A', 'a'])

  # Return empty list when given empty list.
  tknzr.max_seq_len = 2
  assert tknzr.batch_enc(batch_txt=[]) == []

  # Batch encoding format.
  tknzr.max_seq_len = 4
  assert tknzr.batch_enc(batch_txt=['a A', 'A a']) == [
    [
      BOS_TKID,
      tknzr.tk2id['a'],
      tknzr.tk2id['A'],
      EOS_TKID,
    ],
    [
      BOS_TKID,
      tknzr.tk2id['A'],
      tknzr.tk2id['a'],
      EOS_TKID,
    ],
  ]

  # Truncate and pad to specified length.
  tknzr.max_seq_len = 4
  assert tknzr.batch_enc(batch_txt=['a', 'a A', 'a A A']) == [
    [
      BOS_TKID,
      tknzr.tk2id['a'],
      EOS_TKID,
      PAD_TKID,
    ],
    [
      BOS_TKID,
      tknzr.tk2id['a'],
      tknzr.tk2id['A'],
      EOS_TKID,
    ],
    [
      BOS_TKID,
      tknzr.tk2id['a'],
      tknzr.tk2id['A'],
      tknzr.tk2id['A'],
    ],
  ]

  # Unknown tokens.
  tknzr.max_seq_len = 4
  assert tknzr.batch_enc(batch_txt=['a', 'a b', 'a b c']) == [
    [
      BOS_TKID,
      tknzr.tk2id['a'],
      EOS_TKID,
      PAD_TKID,
    ],
    [
      BOS_TKID,
      tknzr.tk2id['a'],
      UNK_TKID,
      EOS_TKID,
    ],
    [
      BOS_TKID,
      tknzr.tk2id['a'],
      UNK_TKID,
      UNK_TKID,
    ],
  ]


def test_uncased_batch_enc() -> None:
  """Encode batch of text to batch of token ids (case-insensitive)."""
  tknzr = WsTknzr(is_uncased=True, max_seq_len=1, max_vocab=-1, min_count=0)
  tknzr.build_vocab(batch_txt=['a'])

  # Return empty list when given empty list.
  tknzr.max_seq_len = 2
  assert tknzr.batch_enc(batch_txt=[]) == []

  # Batch encoding format.
  tknzr.max_seq_len = 4
  assert tknzr.batch_enc(batch_txt=['a A', 'A a']) == [
    [
      BOS_TKID,
      tknzr.tk2id['a'],
      tknzr.tk2id['a'],
      EOS_TKID,
    ],
    [
      BOS_TKID,
      tknzr.tk2id['a'],
      tknzr.tk2id['a'],
      EOS_TKID,
    ],
  ]

  # Truncate and pad to specified length.
  tknzr.max_seq_len = 4
  assert tknzr.batch_enc(batch_txt=['a', 'a A', 'a A A']) == [
    [
      BOS_TKID,
      tknzr.tk2id['a'],
      EOS_TKID,
      PAD_TKID,
    ],
    [
      BOS_TKID,
      tknzr.tk2id['a'],
      tknzr.tk2id['a'],
      EOS_TKID,
    ],
    [
      BOS_TKID,
      tknzr.tk2id['a'],
      tknzr.tk2id['a'],
      tknzr.tk2id['a'],
    ],
  ]

  # Unknown tokens.
  tknzr.max_seq_len = 4
  assert tknzr.batch_enc(batch_txt=['a', 'a b', 'a b c']) == [
    [
      BOS_TKID,
      tknzr.tk2id['a'],
      EOS_TKID,
      PAD_TKID,
    ],
    [
      BOS_TKID,
      tknzr.tk2id['a'],
      UNK_TKID,
      EOS_TKID,
    ],
    [
      BOS_TKID,
      tknzr.tk2id['a'],
      UNK_TKID,
      UNK_TKID,
    ],
  ]


def test_batch_dec() -> None:
  """Decode batch of token ids to batch of text."""
  tknzr = WsTknzr(is_uncased=False, max_seq_len=128, max_vocab=-1, min_count=0)
  tknzr.build_vocab(batch_txt=['A', 'a'])

  # Return empty list when given empty list.
  assert tknzr.batch_dec(batch_tkids=[]) == []

  # Decoding format.
  assert tknzr.batch_dec(
    batch_tkids=[
      [
        BOS_TKID,
        tknzr.tk2id['a'],
        UNK_TKID,
        tknzr.tk2id['A'],
        EOS_TKID,
        PAD_TKID,
      ],
      [
        BOS_TKID,
        UNK_TKID,
        tknzr.tk2id['a'],
        UNK_TKID,
        EOS_TKID,
        PAD_TKID,
      ],
    ],
    rm_sp_tks=False,
  ) == [
    f'{BOS_TK} a {UNK_TK} A {EOS_TK} {PAD_TK}',
    f'{BOS_TK} {UNK_TK} a {UNK_TK} {EOS_TK} {PAD_TK}',
  ]

  # Remove special tokens but not unknown tokens.
  assert tknzr.batch_dec(
    batch_tkids=[
      [
        BOS_TKID,
        tknzr.tk2id['a'],
        UNK_TKID,
        tknzr.tk2id['A'],
        EOS_TKID,
        PAD_TKID,
      ],
      [
        BOS_TKID,
        UNK_TKID,
        tknzr.tk2id['a'],
        UNK_TKID,
        EOS_TKID,
        PAD_TKID,
      ],
    ],
    rm_sp_tks=True,
  ) == [
    f'a {UNK_TK} A',
    f'{UNK_TK} a {UNK_TK}',
  ]

  # Convert unknown id to unknown tokens.
  assert tknzr.batch_dec(
    batch_tkids=[
      [
        BOS_TKID,
        max(tknzr.tk2id.values()) + 1,
        EOS_TKID,
        PAD_TKID,
      ],
      [
        BOS_TKID,
        max(tknzr.tk2id.values()) + 2,
        EOS_TKID,
        PAD_TKID,
      ],
    ],
    rm_sp_tks=False,
  ) == [f'{BOS_TK} {UNK_TK} {EOS_TK} {PAD_TK}', f'{BOS_TK} {UNK_TK} {EOS_TK} {PAD_TK}']
