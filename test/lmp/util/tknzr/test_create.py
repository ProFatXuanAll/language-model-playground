r"""Test construction utilities for all tokenizers.

Test target:
- :py:meth:`lmp.util.tknzr.create`.
"""

import lmp.util.tknzr
from lmp.tknzr import BaseTknzr, CharTknzr, WsTknzr


def test_create_char_tknzr():
    r"""Test construction for character tokenizer."""
    is_uncased = False
    max_vocab = -1
    min_count = 1
    tk2id = {
        BaseTknzr.bos_tk: BaseTknzr.bos_tkid,
        BaseTknzr.eos_tk: BaseTknzr.eos_tkid,
        BaseTknzr.pad_tk: BaseTknzr.pad_tkid,
        BaseTknzr.unk_tk: BaseTknzr.unk_tkid,
        'a': 4,
        'b': 5,
        'c': 6,
    }

    tknzr = lmp.util.tknzr.create(
        is_uncased=is_uncased,
        max_vocab=max_vocab,
        min_count=min_count,
        tknzr_name=CharTknzr.tknzr_name,
        tk2id=tk2id,
    )

    assert isinstance(tknzr, CharTknzr)
    assert tknzr.is_uncased == is_uncased
    assert tknzr.max_vocab == max_vocab
    assert tknzr.min_count == min_count
    assert tknzr.tk2id == tk2id


def test_create_ws_tknzr():
    r"""Test construction for whitespace tokenizer."""
    is_uncased = False
    max_vocab = -1
    min_count = 1
    tk2id = {
        BaseTknzr.bos_tk: BaseTknzr.bos_tkid,
        BaseTknzr.eos_tk: BaseTknzr.eos_tkid,
        BaseTknzr.pad_tk: BaseTknzr.pad_tkid,
        BaseTknzr.unk_tk: BaseTknzr.unk_tkid,
        'a': 4,
        'b': 5,
        'c': 6,
    }

    tknzr = lmp.util.tknzr.create(
        is_uncased=is_uncased,
        max_vocab=max_vocab,
        min_count=min_count,
        tknzr_name=WsTknzr.tknzr_name,
        tk2id=tk2id,
    )

    assert isinstance(tknzr, WsTknzr)
    assert tknzr.is_uncased == is_uncased
    assert tknzr.max_vocab == max_vocab
    assert tknzr.min_count == min_count
    assert tknzr.tk2id == tk2id
