"""Test token encoding and decoding.

Test target:
- :py:meth:`lmp.tknzr.CharTknzr.batch_dec`.
- :py:meth:`lmp.tknzr.CharTknzr.batch_enc`.
"""

from lmp.tknzr import CharTknzr


def test_cased_batch_enc() -> None:
  """Encode batch of text to batch of token ids (case-sensitive)."""
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
  # Return empty list when given empty list.
  assert tknzr.batch_enc([]) == []
  # Batch encoding format.
  assert tknzr.batch_enc(['aA', 'Aa']) == [
    [
      CharTknzr.bos_tkid,
      tk2id['a'],
      tk2id['A'],
      CharTknzr.eos_tkid,
    ],
    [
      CharTknzr.bos_tkid,
      tk2id['A'],
      tk2id['a'],
      CharTknzr.eos_tkid,
    ],
  ]
  # Automatically calculate `max_seq_len`.
  assert tknzr.batch_enc(['a', 'aa', 'aaa']) == [
    [
      CharTknzr.bos_tkid,
      tk2id['a'],
      CharTknzr.eos_tkid,
      CharTknzr.pad_tkid,
      CharTknzr.pad_tkid,
    ],
    [
      CharTknzr.bos_tkid,
      tk2id['a'],
      tk2id['a'],
      CharTknzr.eos_tkid,
      CharTknzr.pad_tkid,
    ],
    [
      CharTknzr.bos_tkid,
      tk2id['a'],
      tk2id['a'],
      tk2id['a'],
      CharTknzr.eos_tkid,
    ],
  ]
  # Truncate and pad to specified length.
  assert tknzr.batch_enc(['a', 'aA', 'aAA'], max_seq_len=4) == [
    [
      CharTknzr.bos_tkid,
      tk2id['a'],
      CharTknzr.eos_tkid,
      CharTknzr.pad_tkid,
    ],
    [
      CharTknzr.bos_tkid,
      tk2id['a'],
      tk2id['A'],
      CharTknzr.eos_tkid,
    ],
    [
      CharTknzr.bos_tkid,
      tk2id['a'],
      tk2id['A'],
      tk2id['A'],
    ],
  ]
  # Unknown tokens.
  assert tknzr.batch_enc(['a', 'ab', 'abc'], max_seq_len=4) == [
    [
      CharTknzr.bos_tkid,
      tk2id['a'],
      CharTknzr.eos_tkid,
      CharTknzr.pad_tkid,
    ],
    [
      CharTknzr.bos_tkid,
      tk2id['a'],
      CharTknzr.unk_tkid,
      CharTknzr.eos_tkid,
    ],
    [
      CharTknzr.bos_tkid,
      tk2id['a'],
      CharTknzr.unk_tkid,
      CharTknzr.unk_tkid,
    ],
  ]


def test_uncased_batch_enc() -> None:
  """Encode batch of text to batch of token ids (case-insensitive)."""
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
  # Return empty list when given empty list.
  assert tknzr.batch_enc([]) == []
  # Batch encoding format.
  assert tknzr.batch_enc(['aA', 'Aa']) == [
    [
      CharTknzr.bos_tkid,
      tk2id['a'],
      tk2id['a'],
      CharTknzr.eos_tkid,
    ],
    [
      CharTknzr.bos_tkid,
      tk2id['a'],
      tk2id['a'],
      CharTknzr.eos_tkid,
    ],
  ]
  # Automatically calculate `max_seq_len`.
  assert tknzr.batch_enc(['a', 'aa', 'aaa']) == [
    [
      CharTknzr.bos_tkid,
      tk2id['a'],
      CharTknzr.eos_tkid,
      CharTknzr.pad_tkid,
      CharTknzr.pad_tkid,
    ],
    [
      CharTknzr.bos_tkid,
      tk2id['a'],
      tk2id['a'],
      CharTknzr.eos_tkid,
      CharTknzr.pad_tkid,
    ],
    [
      CharTknzr.bos_tkid,
      tk2id['a'],
      tk2id['a'],
      tk2id['a'],
      CharTknzr.eos_tkid,
    ],
  ]
  # Truncate and pad to specified length.
  assert tknzr.batch_enc(['a', 'aA', 'aAA'], max_seq_len=4) == [
    [
      CharTknzr.bos_tkid,
      tk2id['a'],
      CharTknzr.eos_tkid,
      CharTknzr.pad_tkid,
    ],
    [
      CharTknzr.bos_tkid,
      tk2id['a'],
      tk2id['a'],
      CharTknzr.eos_tkid,
    ],
    [
      CharTknzr.bos_tkid,
      tk2id['a'],
      tk2id['a'],
      tk2id['a'],
    ],
  ]
  # Unknown tokens.
  assert tknzr.batch_enc(['a', 'ab', 'abc'], max_seq_len=4) == [
    [
      CharTknzr.bos_tkid,
      tk2id['a'],
      CharTknzr.eos_tkid,
      CharTknzr.pad_tkid,
    ],
    [
      CharTknzr.bos_tkid,
      tk2id['a'],
      CharTknzr.unk_tkid,
      CharTknzr.eos_tkid,
    ],
    [
      CharTknzr.bos_tkid,
      tk2id['a'],
      CharTknzr.unk_tkid,
      CharTknzr.unk_tkid,
    ],
  ]


def test_batch_dec() -> None:
  """Decode batch of token ids to batch of text."""
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
  # Return empty list when given empty list.
  assert tknzr.batch_dec([]) == []
  # Decoding format.
  assert tknzr.batch_dec(
    [
      [
        CharTknzr.bos_tkid,
        tk2id['a'],
        CharTknzr.unk_tkid,
        tk2id['A'],
        CharTknzr.eos_tkid,
        CharTknzr.pad_tkid,
      ],
      [
        CharTknzr.bos_tkid,
        CharTknzr.unk_tkid,
        tk2id['a'],
        CharTknzr.unk_tkid,
        CharTknzr.eos_tkid,
        CharTknzr.pad_tkid,
      ],
    ],
    rm_sp_tks=False,
  ) == [
    f'{CharTknzr.bos_tk}a{CharTknzr.unk_tk}A{CharTknzr.eos_tk}{CharTknzr.pad_tk}',
    f'{CharTknzr.bos_tk}{CharTknzr.unk_tk}a{CharTknzr.unk_tk}{CharTknzr.eos_tk}{CharTknzr.pad_tk}',
  ]
  # Remove special tokens but not unknown tokens.
  assert tknzr.batch_dec(
    [
      [
        CharTknzr.bos_tkid,
        tk2id['a'],
        CharTknzr.unk_tkid,
        tk2id['A'],
        CharTknzr.eos_tkid,
        CharTknzr.pad_tkid,
      ],
      [
        CharTknzr.bos_tkid,
        CharTknzr.unk_tkid,
        tk2id['a'],
        CharTknzr.unk_tkid,
        CharTknzr.eos_tkid,
        CharTknzr.pad_tkid,
      ],
    ],
    rm_sp_tks=True,
  ) == [
    f'a{CharTknzr.unk_tk}A',
    f'{CharTknzr.unk_tk}a{CharTknzr.unk_tk}',
  ]
  # Convert unknown id to unknown tokens.
  assert tknzr.batch_dec(
    [
      [
        CharTknzr.bos_tkid,
        max(tk2id.values()) + 1,
        CharTknzr.eos_tkid,
        CharTknzr.pad_tkid,
      ],
      [
        CharTknzr.bos_tkid,
        max(tk2id.values()) + 2,
        CharTknzr.eos_tkid,
        CharTknzr.pad_tkid,
      ],
    ],
    rm_sp_tks=False,
  ) == [
    f'{CharTknzr.bos_tk}{CharTknzr.unk_tk}{CharTknzr.eos_tk}{CharTknzr.pad_tk}',
    f'{CharTknzr.bos_tk}{CharTknzr.unk_tk}{CharTknzr.eos_tk}{CharTknzr.pad_tk}'
  ]
