"""Test token encoding and decoding.

Test target:
- :py:meth:`lmp.tknzr.WsTknzr.batch_dec`.
- :py:meth:`lmp.tknzr.WsTknzr.batch_enc`.
"""

from lmp.tknzr import WsTknzr


def test_cased_batch_enc() -> None:
  """Encode batch of text to batch of token ids (case-sensitive)."""
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
  # Return empty list when given empty list.
  assert tknzr.batch_enc(batch_txt=[], max_seq_len=100) == []
  # Batch encoding format.
  assert tknzr.batch_enc(batch_txt=['a A', 'A a'], max_seq_len=4) == [
    [
      WsTknzr.bos_tkid,
      tk2id['a'],
      tk2id['A'],
      WsTknzr.eos_tkid,
    ],
    [
      WsTknzr.bos_tkid,
      tk2id['A'],
      tk2id['a'],
      WsTknzr.eos_tkid,
    ],
  ]
  # Truncate and pad to specified length.
  assert tknzr.batch_enc(batch_txt=['a', 'a A', 'a A A'], max_seq_len=4) == [
    [
      WsTknzr.bos_tkid,
      tk2id['a'],
      WsTknzr.eos_tkid,
      WsTknzr.pad_tkid,
    ],
    [
      WsTknzr.bos_tkid,
      tk2id['a'],
      tk2id['A'],
      WsTknzr.eos_tkid,
    ],
    [
      WsTknzr.bos_tkid,
      tk2id['a'],
      tk2id['A'],
      tk2id['A'],
    ],
  ]
  # Unknown tokens.
  assert tknzr.batch_enc(batch_txt=['a', 'a b', 'a b c'], max_seq_len=4) == [
    [
      WsTknzr.bos_tkid,
      tk2id['a'],
      WsTknzr.eos_tkid,
      WsTknzr.pad_tkid,
    ],
    [
      WsTknzr.bos_tkid,
      tk2id['a'],
      WsTknzr.unk_tkid,
      WsTknzr.eos_tkid,
    ],
    [
      WsTknzr.bos_tkid,
      tk2id['a'],
      WsTknzr.unk_tkid,
      WsTknzr.unk_tkid,
    ],
  ]


def test_uncased_batch_enc() -> None:
  """Encode batch of text to batch of token ids (case-insensitive)."""
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
  # Return empty list when given empty list.
  assert tknzr.batch_enc(batch_txt=[], max_seq_len=100) == []
  # Batch encoding format.
  assert tknzr.batch_enc(batch_txt=['a A', 'A a'], max_seq_len=4) == [
    [
      WsTknzr.bos_tkid,
      tk2id['a'],
      tk2id['a'],
      WsTknzr.eos_tkid,
    ],
    [
      WsTknzr.bos_tkid,
      tk2id['a'],
      tk2id['a'],
      WsTknzr.eos_tkid,
    ],
  ]
  # Truncate and pad to specified length.
  assert tknzr.batch_enc(batch_txt=['a', 'a A', 'a A A'], max_seq_len=4) == [
    [
      WsTknzr.bos_tkid,
      tk2id['a'],
      WsTknzr.eos_tkid,
      WsTknzr.pad_tkid,
    ],
    [
      WsTknzr.bos_tkid,
      tk2id['a'],
      tk2id['a'],
      WsTknzr.eos_tkid,
    ],
    [
      WsTknzr.bos_tkid,
      tk2id['a'],
      tk2id['a'],
      tk2id['a'],
    ],
  ]
  # Unknown tokens.
  assert tknzr.batch_enc(batch_txt=['a', 'a b', 'a b c'], max_seq_len=4) == [
    [
      WsTknzr.bos_tkid,
      tk2id['a'],
      WsTknzr.eos_tkid,
      WsTknzr.pad_tkid,
    ],
    [
      WsTknzr.bos_tkid,
      tk2id['a'],
      WsTknzr.unk_tkid,
      WsTknzr.eos_tkid,
    ],
    [
      WsTknzr.bos_tkid,
      tk2id['a'],
      WsTknzr.unk_tkid,
      WsTknzr.unk_tkid,
    ],
  ]


def test_batch_dec() -> None:
  """Decode batch of token ids to batch of text."""
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
  # Return empty list when given empty list.
  assert tknzr.batch_dec(batch_tkids=[]) == []
  # Decoding format.
  assert tknzr.batch_dec(
    batch_tkids=[
      [
        WsTknzr.bos_tkid,
        tk2id['a'],
        WsTknzr.unk_tkid,
        tk2id['A'],
        WsTknzr.eos_tkid,
        WsTknzr.pad_tkid,
      ],
      [
        WsTknzr.bos_tkid,
        WsTknzr.unk_tkid,
        tk2id['a'],
        WsTknzr.unk_tkid,
        WsTknzr.eos_tkid,
        WsTknzr.pad_tkid,
      ],
    ],
    rm_sp_tks=False,
  ) == [
    f'{WsTknzr.bos_tk} a {WsTknzr.unk_tk} A {WsTknzr.eos_tk} {WsTknzr.pad_tk}',
    f'{WsTknzr.bos_tk} {WsTknzr.unk_tk} a {WsTknzr.unk_tk} {WsTknzr.eos_tk} {WsTknzr.pad_tk}',
  ]
  # Remove special tokens but not unknown tokens.
  assert tknzr.batch_dec(
    batch_tkids=[
      [
        WsTknzr.bos_tkid,
        tk2id['a'],
        WsTknzr.unk_tkid,
        tk2id['A'],
        WsTknzr.eos_tkid,
        WsTknzr.pad_tkid,
      ],
      [
        WsTknzr.bos_tkid,
        WsTknzr.unk_tkid,
        tk2id['a'],
        WsTknzr.unk_tkid,
        WsTknzr.eos_tkid,
        WsTknzr.pad_tkid,
      ],
    ],
    rm_sp_tks=True,
  ) == [
    f'a {WsTknzr.unk_tk} A',
    f'{WsTknzr.unk_tk} a {WsTknzr.unk_tk}',
  ]
  # Convert unknown id to unknown tokens.
  assert tknzr.batch_dec(
    batch_tkids=[
      [
        WsTknzr.bos_tkid,
        max(tk2id.values()) + 1,
        WsTknzr.eos_tkid,
        WsTknzr.pad_tkid,
      ],
      [
        WsTknzr.bos_tkid,
        max(tk2id.values()) + 2,
        WsTknzr.eos_tkid,
        WsTknzr.pad_tkid,
      ],
    ],
    rm_sp_tks=False,
  ) == [
    f'{WsTknzr.bos_tk} {WsTknzr.unk_tk} {WsTknzr.eos_tk} {WsTknzr.pad_tk}',
    f'{WsTknzr.bos_tk} {WsTknzr.unk_tk} {WsTknzr.eos_tk} {WsTknzr.pad_tk}'
  ]
