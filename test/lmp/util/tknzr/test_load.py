r"""Test loading utilities for all tokenizers.

Test target:
- :py:meth:`lmp.util.tknzr.load`.
"""

import lmp.util.tknzr
from lmp.tknzr import BaseTknzr, CharTknzr, WsTknzr


def test_load_char_tknzr(exp_name: str, clean_tknzr):
    r"""Ensure consistency between creation and load."""
    tknzr = lmp.util.tknzr.create(
        is_uncased=False,
        max_vocab=-1,
        min_count=1,
        tknzr_name=CharTknzr.tknzr_name,
        tk2id={
            BaseTknzr.bos_tk: BaseTknzr.bos_tkid,
            BaseTknzr.eos_tk: BaseTknzr.eos_tkid,
            BaseTknzr.pad_tk: BaseTknzr.pad_tkid,
            BaseTknzr.unk_tk: BaseTknzr.unk_tkid,
            'a': 4,
            'b': 5,
            'c': 6,
        },
    )
    tknzr.save(exp_name=exp_name)

    load_tknzr = lmp.util.tknzr.load(
        exp_name=exp_name,
        tknzr_name=CharTknzr.tknzr_name,
    )

    assert isinstance(load_tknzr, CharTknzr)
    assert load_tknzr.is_uncased == tknzr.is_uncased
    assert load_tknzr.max_vocab == tknzr.max_vocab
    assert load_tknzr.min_count == tknzr.min_count
    assert load_tknzr.tk2id == tknzr.tk2id
    assert load_tknzr.id2tk == tknzr.id2tk


def test_load_ws_tknzr(exp_name: str, clean_tknzr):
    r"""Ensure consistency between creation and load."""
    tknzr = lmp.util.tknzr.create(
        is_uncased=False,
        max_vocab=-1,
        min_count=1,
        tknzr_name=WsTknzr.tknzr_name,
        tk2id={
            BaseTknzr.bos_tk: BaseTknzr.bos_tkid,
            BaseTknzr.eos_tk: BaseTknzr.eos_tkid,
            BaseTknzr.pad_tk: BaseTknzr.pad_tkid,
            BaseTknzr.unk_tk: BaseTknzr.unk_tkid,
            'a': 4,
            'b': 5,
            'c': 6,
        },
    )
    tknzr.save(exp_name=exp_name)

    load_tknzr = lmp.util.tknzr.load(
        exp_name=exp_name,
        tknzr_name=WsTknzr.tknzr_name,
    )

    assert isinstance(load_tknzr, WsTknzr)
    assert load_tknzr.is_uncased == tknzr.is_uncased
    assert load_tknzr.max_vocab == tknzr.max_vocab
    assert load_tknzr.min_count == tknzr.min_count
    assert load_tknzr.tk2id == tknzr.tk2id
    assert load_tknzr.id2tk == tknzr.id2tk
