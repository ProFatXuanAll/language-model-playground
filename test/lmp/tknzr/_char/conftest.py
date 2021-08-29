r"""Setup fixtures for testing :py:class:`lmp.tknzr.CharTknzr`."""

import pytest

from lmp.tknzr import CharTknzr


@pytest.fixture
def char_tknzr() -> CharTknzr:
    r"""Common setup of character tokenizer."""

    return CharTknzr(
        is_uncased=True,
        max_vocab=-1,
        min_count=1,
        tk2id={
            CharTknzr.bos_tk: CharTknzr.bos_tkid,
            CharTknzr.eos_tk: CharTknzr.eos_tkid,
            CharTknzr.pad_tk: CharTknzr.pad_tkid,
            CharTknzr.unk_tk: CharTknzr.unk_tkid,
            'a': 4,
            'b': 5,
            'c': 6,
        },
    )
