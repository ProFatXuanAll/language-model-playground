r"""Setup fixtures for testing :py:class:`lmp.tknzr.WsTknzr`."""

import pytest

from lmp.tknzr import WsTknzr


@pytest.fixture
def ws_tknzr() -> WsTknzr:
    r"""Common setup of whitespace tokenizer."""

    return WsTknzr(
        is_uncased=True,
        max_vocab=-1,
        min_count=1,
        tk2id={
            WsTknzr.bos_tk: WsTknzr.bos_tkid,
            WsTknzr.eos_tk: WsTknzr.eos_tkid,
            WsTknzr.pad_tk: WsTknzr.pad_tkid,
            WsTknzr.unk_tk: WsTknzr.unk_tkid,
            'a': 4,
            'b': 5,
            'c': 6,
        },
    )
